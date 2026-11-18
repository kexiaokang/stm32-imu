"""
W7-A 讲解用：把"一个窗口"和"19 维特征"摊开给你看
=====================================================================
不做训练，只打印真实数字。用来搞清楚三件事：
  1. 一个窗口到底长什么样（128 行 × 6 列）
  2. 19 维特征是怎么从 768 个数字里算出来的
  3. "15 倍数据"到底是什么
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_baseline import (
    NPZ, recover_groups, extract_features, FEATURE_NAMES, moving_avg,
)

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
AXIS_MEANING = {
    "ax": "加速度 X", "ay": "加速度 Y", "az": "加速度 Z（竖直，含重力）",
    "gx": "角速度 X", "gy": "角速度 Y", "gz": "角速度 Z（绕垂直轴转）",
}


def show_window(X, i, title):
    w = X[i].astype(np.int64)
    print(f"\n--- {title} ---")
    print(f"X[{i}] 的形状 = {w.shape}   ← 128 个采样点 × 6 个轴")
    print(f"  一共 {w.size} 个数字\n")
    hdr = "".join(f"{a:>9}" for a in AXES)
    print(f"{'点号':>6}{hdr}")
    for k in (0, 1, 2, 63, 64, 126, 127):
        row = "".join(f"{v:>9d}" for v in w[k])
        print(f"{k:>6}{row}")
    print(f"{'...':>6}")
    return w


def explain_features(X, y, i, classes):
    w = X[i].astype(np.float64)
    print(f"\n--- 把 X[{i}] 的 768 个数字压成 19 个 ---")
    print(f"真实类别: {classes[y[i]]}\n")
    print(f"{'轴':<6}{'含义':<26}{'span':>10}{'mean':>10}{'std':>10}")
    print("-" * 64)
    for c, a in enumerate(AXES):
        s = moving_avg(w[:, c])
        span = s.max() - s.min()
        print(f"{a:<6}{AXIS_MEANING[a]:<26}{span:>10.0f}{w[:, c].mean():>10.0f}{w[:, c].std():>10.0f}")
    neg = float((w[:, 2] < 0).mean())
    print("-" * 64)
    print(f"{'az_neg_ratio':<32}{neg:>32.3f}   ← az < 0 的点占 {neg*100:.0f}%")
    print()
    print("所以 19 个数字 = 6 个轴 × (span + mean + std) = 18，再加 az_neg_ratio = 19")


def group_sizes(X, y, g):
    print("\n--- \"15 倍数据\"到底怎么来的 ---")
    print(f"{'组':>4}{'类别':>14}{'窗口数':>8}")
    print("-" * 30)
    for k in range(int(g.max()) + 1):
        m = g == k
        cls = classes_name[y[m][0]]
        print(f"{k:>4}{cls:>14}{int(m.sum()):>8}")
    print("-" * 30)
    print(f"合计 {len(y)} 个窗口，分 {int(g.max())+1} 组")


if __name__ == "__main__":
    d = np.load(NPZ, allow_pickle=True)
    X, y = d["X"], d["y"].astype(np.int32)
    classes = [str(c) for c in d["classes"]]
    classes_name = {i: c for i, c in enumerate(classes)}
    g = recover_groups(X, y)

    print("=" * 72)
    print("你问的三个东西：1024×6、15 倍、19 维")
    print("=" * 72)

    # 找两个代表性窗口：静置 和 旋转
    i_static = int(np.where(y == 0)[0][0])
    i_rotate = int(np.where(y == 4)[0][0])

    show_window(X, i_static, f"静置类的一个窗口（原始 768 个数字）")
    explain_features(X, y, i_static, classes)
    print("\n" + "=" * 72)
    show_window(X, i_rotate, f"旋转类的一个窗口（原始 768 个数字）")
    explain_features(X, y, i_rotate, classes)
    print("\n" + "=" * 72)

    with np.errstate(all="ignore"):
        pass
    group_sizes(X, y, g)

    print("\n--- 特征矩阵最后长这样 ---")
    F = extract_features(X)
    print(f"F.shape = {F.shape}   ← 421 行（每个窗口一行）× 19 列（19 个特征）")
    print(f"{'行':>4}" + "".join(f"{n[:8]:>10}" for n in FEATURE_NAMES[:6]) + "   ...（共19列）")
    for i in range(3):
        print(f"{i:>4}" + "".join(f"{F[i, k]:>10.0f}" for k in range(6)))
    print(f"标签 y = {y[:10].tolist()} ...（每行对应一个类号）")
