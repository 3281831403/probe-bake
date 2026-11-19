# light-probe

## 题面

室内场景接入光照探针后，墙角开始漏光：当前四面体查找对全部探针组合做线性扫描，10 万探针时直接卡死；重心坐标计算后没有归一，远点颜色饱和；外部点也被外推，墙角漏光；最近探针搜索不跳过 invalid，黑洞探针被复用；遮挡权重完全不看法线，背向探针仍贡献 50%；漏光评分不做钳制，能返回 3.2；修复泄漏只把探针标成无效却不改权重，画面仍然漏；批量烘焙遇到坏网格只跳过，输出缺块；探针数组顺序依赖 dict 迭代，同一场景两次插值结果不同；颜色不钳制到 0..1，HDR 值进入 8 bit 通道后回绕。probe/interpolate.py 是核心，tests/test_interpolate.py 15个红。要修的是：四面体加速查找、重心坐标归一、内部插值不外推、跳过失效探针、按法线遮挡、漏光评分钳制、泄漏修复权重、批量烘焙回滚、消除 dict 顺序依赖、颜色钳制。

## 运行

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态必须恰好为 `15 failed`，且没有 collection/import/syntax error；输出尾部形如：

```text
15 failed in 0.10s
```

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 约束

`tests/test_interpolate.py` 不可修改，评测以该文件为准。实现只允许使用 Python 标准库（math、itertools）和 pytest，离线运行，不使用随机数。修复目标是 `probe/interpolate.py` 中的 8 个函数；`probe` 包、`tests` 目录和本 README 之外不引入额外依赖。
