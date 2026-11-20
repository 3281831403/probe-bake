"""Light-probe interpolation toolkit.

Every probe is a mapping with this shape::

    {"pos": [x, y, z], "color": [r, g, b], "valid": bool}

Positions are finite 3-D world coordinates. Colors are linear RGB values
nominally in ``[0.0, 1.0]``; comparisons use an absolute tolerance of
``1e-6``. An invalid probe keeps its data but must be ignored by every
query.
"""

import itertools
import math

import numpy as np

try:  # Fast path: Qhull-backed Delaunay tetrahedralisation.
    from scipy.spatial import Delaunay, QhullError
except Exception:  # pragma: no cover - scipy is expected but optional
    Delaunay = None

    class QhullError(Exception):
        pass


__all__ = [
    "ProbeError",
    "locate_tetrahedron",
    "barycentric",
    "interpolate_color",
    "nearest_valid",
    "occlusion_weight",
    "leak_score",
    "fix_leaks",
    "bake_batch",
]

_TOL = 1e-6


class ProbeError(Exception):
    """Raised when probe data or a query cannot produce a valid result."""


def _valid_indices(probes):
    return [i for i, probe in enumerate(probes) if probe["valid"]]


def _bary_coords(p0, p1, p2, p3, point):
    """Barycentric weights of ``point`` for the tetrahedron p0..p3.

    Raises ``ProbeError`` for a degenerate tetrahedron. Weights are signed
    and sum to one by construction.
    """
    matrix = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    try:
        w123 = np.linalg.solve(matrix, point - p0)
    except np.linalg.LinAlgError:
        raise ProbeError("degenerate tetrahedron")
    return [1.0 - float(w123.sum()), float(w123[0]), float(w123[1]), float(w123[2])]


def _locate_fallback(positions, point):
    """Locate a containing tetrahedron without scipy.

    Enumerates combinations of an expanding set of nearest neighbours, so
    the cost stays far below the naive O(n**4) scan.
    """
    dists = np.linalg.norm(positions - point, axis=1)
    order = np.argsort(dists, kind="stable")
    n = len(positions)
    k = 4
    while True:
        k = min(n, max(k, 8))
        candidates = order[:k].tolist()
        for combo in itertools.combinations(candidates, 4):
            p0, p1, p2, p3 = (positions[j] for j in combo)
            try:
                weights = _bary_coords(p0, p1, p2, p3, point)
            except ProbeError:
                continue
            if min(weights) >= -_TOL:
                return combo
        if k >= n or k >= 64:
            break
        k *= 2
    raise ProbeError("point lies outside every valid tetrahedron")


def locate_tetrahedron(probes, point):
    """Return indices of four probes forming a tetrahedron containing point.

    Only probes whose ``valid`` flag is true are candidates. Containment
    includes the boundary: every barycentric coordinate may be as low as
    ``-1e-6``. The returned value is a 4-tuple of distinct indices into
    ``probes`` and must be deterministic for equal input.

    The search must not enumerate all ``O(n**4)`` combinations; a cloud of
    100_000 probes is expected to resolve quickly.

    Raise ``ProbeError`` when fewer than four valid probes exist or when
    ``point`` lies outside every valid tetrahedron.
    """
    valid = _valid_indices(probes)
    if len(valid) < 4:
        raise ProbeError("fewer than four valid probes")
    point = np.asarray(point, dtype=float)
    positions = np.array([probes[i]["pos"] for i in valid], dtype=float)

    if Delaunay is None:
        combo = _locate_fallback(positions, point)
        return tuple(valid[j] for j in combo)

    # Expanding local search: tetrahedralise only the k nearest valid
    # probes, growing k until the point lands inside the local hull. A
    # containing tetrahedron exists among the candidates exactly when the
    # point lies in their convex hull, so this is exact while avoiding a
    # costly global tetrahedralisation of the whole cloud.
    dists = np.linalg.norm(positions - point, axis=1)
    order = np.argsort(dists, kind="stable")
    n = len(valid)
    k = min(n, 16)
    while True:
        candidates = order[:k]
        try:
            tri = Delaunay(positions[candidates])
        except QhullError:
            tri = None
        if tri is not None:
            simplex = int(tri.find_simplex(point, tol=_TOL))
            if simplex >= 0:
                combo = tuple(int(candidates[j]) for j in tri.simplices[simplex])
                weights = _bary_coords(*(positions[j] for j in combo), point)
                if min(weights) >= -_TOL:
                    return tuple(valid[j] for j in combo)
        if k >= n:
            raise ProbeError("point lies outside every valid tetrahedron")
        k = min(n, k * 4)


def barycentric(tetra, point):
    """Return ``[w0, w1, w2, w3]`` for point in tetra.

    ``tetra`` is a sequence of four probe mappings. The weights satisfy
    ``sum(weights) == 1`` within an absolute tolerance of ``1e-6`` and
    reconstruct the point::

        point == sum(wi * tetra[i]["pos"] for i in range(4))

    Points outside the tetrahedron produce at least one negative weight;
    weights are signed and are neither clamped nor renormalised here.
    Raise ``ProbeError`` for a wrong-length or degenerate tetrahedron.
    """
    if len(tetra) != 4:
        raise ProbeError("tetrahedron must have exactly four probes")
    p0, p1, p2, p3 = (np.asarray(probe["pos"], dtype=float) for probe in tetra)
    point = np.asarray(point, dtype=float)
    return _bary_coords(p0, p1, p2, p3, point)


def interpolate_color(probes, point):
    """Return the interpolated linear RGB color at ``point`` as a list.

    Only valid probes contribute. An empty or all-invalid probe cloud
    returns black, ``[0.0, 0.0, 0.0]``. When a containing tetrahedron is
    found, blend its vertex colors with barycentric weights clamped at
    zero and renormalised, so no signed extrapolation can occur. A point
    outside the valid hull falls back to the color of ``nearest_valid``
    rather than extrapolating.

    Every returned channel is clamped to ``[0.0, 1.0]``. RGB comparisons
    use an absolute tolerance of ``1e-6``.
    """
    if not any(probe["valid"] for probe in probes):
        return [0.0, 0.0, 0.0]
    try:
        indices = locate_tetrahedron(probes, point)
        weights = barycentric([probes[i] for i in indices], point)
        weights = [max(0.0, w) for w in weights]
        total = sum(weights)
        if total > 0.0:
            color = [
                sum(
                    weights[k] * probes[indices[k]]["color"][channel]
                    for k in range(4)
                )
                / total
                for channel in range(3)
            ]
        else:
            color = list(probes[nearest_valid(probes, point)]["color"])
    except ProbeError:
        color = list(probes[nearest_valid(probes, point)]["color"])
    return [min(1.0, max(0.0, channel)) for channel in color]


def nearest_valid(probes, point):
    """Return the index of the valid probe closest to ``point``.

    Distance is Euclidean. Probes with ``valid`` false are skipped. Ties
    are resolved by the lowest index. Raise ``ProbeError`` when there is no
    valid probe.
    """
    best_index = None
    best_dist = math.inf
    px, py, pz = point
    for i, probe in enumerate(probes):
        if not probe["valid"]:
            continue
        dx = probe["pos"][0] - px
        dy = probe["pos"][1] - py
        dz = probe["pos"][2] - pz
        dist = dx * dx + dy * dy + dz * dz
        if dist < best_dist:
            best_dist = dist
            best_index = i
    if best_index is None:
        raise ProbeError("no valid probe")
    return best_index


def occlusion_weight(probes, point, normal):
    """Return a normal-facing occlusion weight in ``[0.0, 1.0]``.

    The normal is normalised internally; a zero normal raises
    ``ProbeError``. For each valid probe, let ``d`` be the unit direction
    from ``point`` to the probe position and compute
    ``cos = max(0, dot(normal_unit, d))``. A probe coincident with the
    point contributes zero. The result is::

        min(1.0, sum(cos) / number_of_valid_probes)

    Thus a probe behind the surface contributes exactly zero, and no valid
    probes yields ``0.0``. Comparisons use an absolute tolerance of
    ``1e-6``.
    """
    norm = math.sqrt(sum(component * component for component in normal))
    if norm == 0.0:
        raise ProbeError("zero normal")
    unit = [component / norm for component in normal]

    total = 0.0
    count = 0
    for probe in probes:
        if not probe["valid"]:
            continue
        count += 1
        d = [probe["pos"][axis] - point[axis] for axis in range(3)]
        length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        if length == 0.0:
            continue
        cos = sum(unit[axis] * d[axis] for axis in range(3)) / length
        total += max(0.0, cos)
    if count == 0:
        return 0.0
    return min(1.0, total / count)


def leak_score(probes, point, max_distance):
    """Return a leak score in ``[0.0, 1.0]`` for invalid probes near point.

    Each invalid probe at distance ``d < max_distance`` contributes
    ``1.0 - d / max_distance``. The contributions are summed and the final
    score is clamped to ``[0.0, 1.0]``; valid probes do not contribute.
    ``max_distance`` must be positive, otherwise ``ProbeError`` is raised.
    """
    if max_distance <= 0:
        raise ProbeError("max_distance must be positive")
    score = 0.0
    for probe in probes:
        if probe["valid"]:
            continue
        d = math.dist(probe["pos"], point)
        if d < max_distance:
            score += 1.0 - d / max_distance
    return min(1.0, max(0.0, score))


def fix_leaks(probes, threshold):
    """Invalidate and blacken valid probes whose color exceeds threshold.

    For every probe with ``valid`` true whose largest RGB channel is
    greater than ``threshold`` (within the usual ``1e-6`` tolerance), set
    ``probes[i]["valid"] = False`` and
    ``probes[i]["color"] = [0.0, 0.0, 0.0]``. The input list is mutated in
    place, which removes the probe's contribution as well as its stored
    color.

    Return ``{"fixed": [ascending indices], "count": len(fixed)}``.
    ``threshold < 0`` raises ``ProbeError``.
    """
    if threshold < 0:
        raise ProbeError("threshold must be non-negative")
    fixed = []
    for i, probe in enumerate(probes):
        if not probe["valid"]:
            continue
        if max(probe["color"]) > threshold + _TOL:
            probe["valid"] = False
            probe["color"] = [0.0, 0.0, 0.0]
            fixed.append(i)
    return {"fixed": fixed, "count": len(fixed)}


def bake_batch(grids, samples):
    """Bake every named probe grid at its sample points atomically.

    ``grids`` maps a grid name to a probe list; ``samples`` maps the same
    names to lists of 3-D query points. All grids and key sets are
    validated before any baking. A grid is illegal when its name is missing
    from ``samples``, its probe list is empty, or it has no valid probe.

    On any illegal grid or missing sample, raise ``ProbeError`` and return
    nothing: the batch is rolled back as a whole and no partial block is
    produced. Inputs are not modified on failure. On success return::

        {name: [rgb, ...], ...}

    preserving the insertion order of ``grids`` and the order of each
    sample list. Channels are clamped to ``[0.0, 1.0]``.
    """
    for name, probe_list in grids.items():
        if name not in samples:
            raise ProbeError(f"missing samples for grid {name!r}")
        if not probe_list:
            raise ProbeError(f"grid {name!r} has no probes")
        if not any(probe["valid"] for probe in probe_list):
            raise ProbeError(f"grid {name!r} has no valid probe")
    return {
        name: [interpolate_color(probe_list, point) for point in samples[name]]
        for name, probe_list in grids.items()
    }
