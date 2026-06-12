"""结构学习评估指标：SHD、F1 等。

用于对比学习到的图结构与真实图 (ground truth)。
"""


def compute_shd(
    candidate_edges: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
) -> int:
    """计算结构汉明距离 (Structural Hamming Distance)。

    SHD = 两图边的对称差数量 = 多余边 + 缺失边 + 反向边。
    """
    return len(candidate_edges ^ gt_edges)


def compute_f1(
    candidate_edges: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
) -> float:
    """计算 F1 分数。

    F1 = 2 * P * R / (P + R)
    其中 P = TP / (TP + FP), R = TP / (TP + FN)。
    两图均无边时返回 1.0。
    """
    tp = len(candidate_edges & gt_edges)
    fp = len(candidate_edges - gt_edges)
    fn = len(gt_edges - candidate_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
