"""Executable contract for the light-probe interpolation exercise.

Every test currently fails because every public function in
``probe.interpolate`` is a stub. Do not modify this file; repair the
implementation instead.
"""

import itertools

import pytest

from probe.interpolate import (
    ProbeError,
    bake_batch,
    barycentric,
    fix_leaks,
    interpolate_color,
    leak_score,
    locate_tetrahedron,
    nearest_valid,
    occlusion_weight,
)


def _probe(pos, color=(0.5, 0.5, 0.5), valid=True):
    return {
        "pos": [float(value) for value in pos],
        "color": [float(value) for value in color],
        "valid": bool(valid),
    }


def _unit_tetra():
    """Four probes forming the tetrahedron (0,0,0), (1,0,0), (0,1,0), (0,0,1)."""
    return [
        _probe((0.0, 0.0, 0.0), (0.1, 0.2, 0.3)),
        _probe((1.0, 0.0, 0.0), (0.9, 0.1, 0.1)),
        _probe((0.0, 1.0, 0.0), (0.1, 0.9, 0.1)),
        _probe((0.0, 0.0, 1.0), (0.1, 0.1, 0.9)),
    ]


def test_locate_tetrahedron_scales_to_100k_probes():
    extra = (
        _probe((10.0 + i % 97, 10.0 + (i // 97) % 97, 10.0 + i // 9409))
        for i in itertools.islice(itertools.count(), 99_996)
    )
    probes = _unit_tetra() + list(extra)
    assert len(probes) == 100_000

    point = [0.2, 0.2, 0.2]
    indices = locate_tetrahedron(probes, point)
    assert len(indices) == 4
    assert len(set(indices)) == 4
    assert all(probes[index]["valid"] for index in indices)

    weights = barycentric([probes[index] for index in indices], point)
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    assert min(weights) >= -1e-6


def test_barycentric_normalizes_far_outside_point():
    weights = barycentric(_unit_tetra(), [-10.0, -10.0, -10.0])

    assert len(weights) == 4
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    assert min(weights) < 0.0


def test_interpolate_color_does_not_extrapolate_outside_hull():
    tetra = _unit_tetra()

    color = interpolate_color(tetra, [-10.0, -10.0, -10.0])

    # (-10,-10,-10) is outside the hull. The nearest valid probe is the
    # origin, so the result must be its color, never a signed extrapolation.
    assert color == pytest.approx(tetra[0]["color"], abs=1e-6)


def test_nearest_valid_skips_invalid_probe():
    probes = [
        _probe((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), valid=False),
        _probe((2.0, 0.0, 0.0), (0.2, 0.3, 0.4)),
        _probe((5.0, 0.0, 0.0), (0.9, 0.9, 0.9)),
    ]

    assert nearest_valid(probes, [0.1, 0.0, 0.0]) == 1


def test_occlusion_weight_ignores_backfacing_probes():
    normal = [0.0, 0.0, 1.0]
    behind = [
        _probe((0.0, 0.0, -1.0)),
        _probe((1.0, 1.0, -1.0)),
    ]

    assert occlusion_weight(behind, [0.0, 0.0, 0.0], normal) == pytest.approx(
        0.0, abs=1e-6
    )

    front = [_probe((0.0, 0.0, 1.0))]
    assert occlusion_weight(front, [0.0, 0.0, 0.0], normal) == pytest.approx(
        1.0, abs=1e-6
    )


def test_leak_score_is_clamped_to_unit_interval():
    probes = [
        _probe((distance, 0.0, 0.0), valid=False)
        for distance in (0.0, 0.1, 0.2, 0.3)
    ]

    score = leak_score(probes, [0.0, 0.0, 0.0], 1.0)

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0, abs=1e-6)


def test_fix_leaks_clears_color_of_flagged_probe():
    probes = [
        _probe((0.0, 0.0, 0.0), (2.0, 0.2, 0.2)),
        _probe((1.0, 0.0, 0.0), (0.4, 0.4, 0.4)),
    ]
    untouched = list(probes[1]["color"])

    report = fix_leaks(probes, 1.0)

    assert report["fixed"] == [0]
    assert report["count"] == 1
    assert probes[0]["valid"] is False
    assert probes[0]["color"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)
    assert probes[1]["valid"] is True
    assert probes[1]["color"] == pytest.approx(untouched, abs=1e-6)


def test_bake_batch_rolls_back_on_invalid_grid():
    good = _unit_tetra()
    snapshot = [
        {
            "pos": list(probe["pos"]),
            "color": list(probe["color"]),
            "valid": probe["valid"],
        }
        for probe in good
    ]
    grids = {"good": good, "bad": []}
    samples = {"good": [[0.1, 0.1, 0.1]], "bad": [[0.0, 0.0, 0.0]]}

    with pytest.raises(ProbeError):
        bake_batch(grids, samples)

    # No block may be emitted from the good grid, and the inputs must not be
    # partially rewritten before the bad grid is discovered.
    assert good == snapshot


def test_interpolation_is_independent_of_dict_and_probe_order():
    probes = _unit_tetra()
    reordered = [
        {
            "valid": probe["valid"],
            "color": list(probe["color"]),
            "pos": list(probe["pos"]),
        }
        for probe in reversed(probes)
    ]
    point = [0.2, 0.2, 0.2]

    first = interpolate_color(probes, point)
    second = interpolate_color(reordered, point)

    assert first == pytest.approx(second, abs=1e-6)


def test_interpolate_color_clamps_hdr_channels():
    corners = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    probes = [_probe(corner, (2.5, -1.0, 0.5)) for corner in corners]

    color = interpolate_color(probes, [0.2, 0.2, 0.2])

    assert color == pytest.approx([1.0, 0.0, 0.5], abs=1e-6)


def test_interpolate_color_at_probe_position_returns_its_color():
    tetra = _unit_tetra()

    color = interpolate_color(tetra, [0.0, 0.0, 0.0])

    assert color == pytest.approx(tetra[0]["color"], abs=1e-6)


def test_interpolate_color_with_empty_probes_returns_black():
    color = interpolate_color([], [0.0, 0.0, 0.0])

    assert color == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_fix_leaks_rejects_negative_threshold():
    with pytest.raises(ProbeError):
        fix_leaks(_unit_tetra(), -0.1)


def test_barycentric_weights_sum_to_one():
    weights = barycentric(_unit_tetra(), [0.2, 0.15, 0.1])

    assert len(weights) == 4
    assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    assert all(weight >= -1e-6 for weight in weights)


def test_bake_batch_preserves_grid_order_on_success():
    grids = {"alpha": _unit_tetra(), "beta": _unit_tetra()}
    samples = {
        "alpha": [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]],
        "beta": [[0.3, 0.3, 0.3]],
    }

    baked = bake_batch(grids, samples)

    assert list(baked) == ["alpha", "beta"]
    assert len(baked["alpha"]) == 2
    assert len(baked["beta"]) == 1
    for name in ("alpha", "beta"):
        for actual, point in zip(baked[name], samples[name]):
            assert actual == pytest.approx(
                interpolate_color(grids[name], point), abs=1e-6
            )
