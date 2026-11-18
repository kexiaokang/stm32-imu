"""
W7-A：PC 端基线训练（随机森林）
=====================================================================
目的：回答一个问题——**421 个窗口，够不够把 5 类分开？**

为什么先训随机森林，不上神经网络：
  1. 421 个样本太少，神经网络会直接过拟合；随机森林在这种规模上更稳。
  2. 它有 feature_importances_，能告诉你"哪几个特征真的有用"——
     这几个特征就是后面部署到板子时要算的东西。
  3. 训练 3 秒，改一行就能重跑。W9 再上 TFLite。

⚠️ 这一步最容易犯的错：**随机切分会导致准确率虚高**
   窗口是 128 点 / 步长 64（50% 重叠），同一轮里相邻两个窗口有 64 个点完全一样。
   如果随机打乱切分，同一轮的窗口会同时出现在训练集和测试集里，
   模型等于是"背答案"，准确率能虚高 10–20 个点。
   所以这里用 **GroupKFold，按"轮"分组**：整轮不进训练集，就整轮不进测试集。

   但 dataset.npz 里没存轮次编号，怎么办？
   —— 用重叠关系反推：如果窗口 i 的后 64 点 == 窗口 i+1 的前 64 点，
      说明它们是同一轮里相邻的两个窗口。见 recover_groups()。
"""

import sys
from pathlib import Path

import numpy as np

# sklearn 1.0.2（老版本，但和 numpy 1.26 匹配）
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix

# clean_data.py 里的 moving_avg / WIN / STRIDE 直接复用，
# 保证"判据"和 W6-A 清洗时用的是同一个算法（10 点滑动平均，不用 convolve）
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clean_data import moving_avg, WIN, STRIDE

NPZ = HERE / "dataset.npz"
SMOOTH = 10
STATIC_T = 800.0     # 和 clean_data.py 一致


# ---------------------------------------------------------------- 分组反推
def recover_groups(X, y):
    """从 50% 重叠关系反推每个窗口所属的采集轮次，用于按轮分组。

    判据：X[i] 的后半段（第 STRIDE~WIN-1 点）与 X[i+1] 的前半段
    （第 0~STRIDE-1 点）逐点相等，且两者类别相同，则视为同一轮中相邻的
    两个窗口；否则轮次编号自增。GroupKFold 据此保证整轮只落在一侧，
    避免同轮窗口同时进入训练集和测试集而导致准确率虚高。

    Args:
        X: 窗口数据，形状为 (N, WIN, C) 的 ndarray。
        y: 每个窗口的类别标签，形状为 (N,) 的 ndarray。

    Returns:
        每个窗口所属的轮次编号，形状为 (N,) 的 int32 数组，从 0 开始连续递增。
    """
    n = len(X)
    g = np.zeros(n, dtype=np.int32)   # 第 i 个窗口所属的轮次编号，从 0 开始
    cur = 0                           # 当前正在累积的轮次编号
    for i in range(n - 1):
        # 判据：类别相同 + 后 STRIDE 点与前 STRIDE 点逐点相等 → 同一轮的相邻窗口
        same_round = (y[i] == y[i + 1]) and np.array_equal(X[i][STRIDE:], X[i + 1][:-STRIDE])
        if not same_round:
            cur += 1                  # 重叠关系断了，说明进入了新的一轮
        g[i + 1] = cur                # 若相邻则沿用 cur，否则用新的 cur
    return g


# ---------------------------------------------------------------- 特征
# 为什么不用原始的 128×6=768 维？
#   421 个样本 vs 768 维特征，比例失衡（1:1.8），必然过拟合。
#   而且板子上也算不动。
# 所以用 19 个"物理意义明确"的统计特征，和 W5-A 那五条判别规则一一对应。
FEATURE_NAMES = []
for _ax in ("ax", "ay", "az", "gx", "gy", "gz"):
    FEATURE_NAMES += [f"{_ax}_span", f"{_ax}_mean", f"{_ax}_std"]
FEATURE_NAMES.append("az_neg_ratio")


def extract_features(X):
    """X=(N,128,6) → F=(N,19)

    每个轴 3 个：平滑跨度（≈低频幅度）、均值（≈重力分量）、标准差（≈能量）
    再加 1 个：az 负值比例（区分"翻面"的唯一可靠特征，flip 时重力反向）
    """
    n = len(X)                                                 # 窗口总数 N
    F = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)    # 输出特征矩阵 (N, 19)
    for i in range(n):                                         # 逐个窗口提取特征
        w = X[i].astype(np.float64)                            # 第 i 个窗口 (128, 6)，转 float 便于统计
        k = 0                                                  # 当前待写入的特征列下标
        for col in range(6):                                   # 遍历 6 个轴：ax ay az gx gy gz
            s = moving_avg(w[:, col])                          # 10 点滑动平均 ≈ 10Hz 低通，压低高频抖动
            F[i, k] = s.max() - s.min();  k += 1               # 平滑跨度 = 平滑后极差，反映低频摆动幅度
            F[i, k] = w[:, col].mean();   k += 1               # 均值 ≈ 重力/角速度的静态分量
            F[i, k] = w[:, col].std();    k += 1               # 标准差 ≈ 动态能量，区分"静止"与"晃动"
        F[i, k] = float((w[:, 2] < 0).mean())                  # az<0 的采样点占比：翻面时重力反向，此值跳变
    return F                                                   # (N, 19)，列顺序与 FEATURE_NAMES 一致


# ---------------------------------------------------------------- 规则基线
def rule_predict(X):
    """W5-A 里那五条规则，不用训练，直接判。用来验证"数据本身是否可分"。"""
    out = np.zeros(len(X), dtype=np.int32)
    for i in range(len(X)):
        w = X[i].astype(np.float64)
        spans = []
        for c in (0, 1, 2, 5):                 # ax ay az gz
            s = moving_avg(w[:, c])
            spans.append(s.max() - s.min())
        ax, ay, az, gz = spans
        az_neg = (w[:, 2] < 0).mean()

        if gz > 40000:                          # 绕垂直轴转
            p = 4
        elif az_neg > 0.20:                     # 重力反向 = 翻面
            p = 3
        elif max(spans) < 2000:                 # 四轴全都很平 = 静置
            p = 0
        elif az < 9000 and ax > 3000:           # 水平推
            p = 2
        else:                                   # 剩下 = 拿起
            p = 1
        out[i] = p
    return out


# ---------------------------------------------------------------- 主流程
def main():
    d = np.load(NPZ, allow_pickle=True)
    X, y = d["X"], d["y"].astype(np.int32)
    classes = [str(c) for c in d["classes"]]
    n_cls = len(classes)

    print("=" * 70)
    print("W7-A  PC 端基线训练")
    print("=" * 70)
    print(f"数据: {NPZ}")
    print(f"X = {X.shape} {X.dtype}   y = {y.shape}   类别 {n_cls}")
    print(f"分布: {np.bincount(y, minlength=n_cls).tolist()}  -> {classes}")
    print()

    # ---- 1. 反推轮次 ----
    g = recover_groups(X, y)
    n_groups = int(g.max()) + 1
    print(f"[1] 反推分组: 共 {n_groups} 组")
    print("    预期 29 轮；实际更多是正常的——被剔除的窗口会在轮中间断开，")
    print("    断点处会多分出一组。组数 >= 轮数就说明分组是安全的（只会更严格，不会泄露）。")
    per = []
    for c in range(n_cls):
        per.append(len(np.unique(g[y == c])))
    print(f"    每类组数: {per}   （0_static 剔除 0 / pickup 1 / shake 0 / flip 0 / rotate 9）")
    if n_groups <= 5:
        print("    !! 轮数太少，GroupKFold 无法切 5 折")
        return
    print()

    # ---- 2. 特征 ----
    F = extract_features(X)
    print(f"[2] 特征: {F.shape[0]} 样本 × {F.shape[1]} 维")
    print()

    # ---- 3. 规则基线（不训练）----
    rule_pred = rule_predict(X)
    rule_acc = accuracy_score(y, rule_pred)
    print(f"[3] 规则基线（W5-A 五条手写规则，0 训练）: 准确率 {rule_acc*100:.1f}%")
    print("    ! 注意：那五条阈值是在【整轮 1024 点】的统计上定的，")
    print("      直接套到【128 点窗口】上会掉一截——这不是数据变差，是统计口径变了。")
    print("      它的作用是说明：同一个判据换个尺度就不成立，所以不能只靠手写规则。")
    print()

    # ---- 4. 随机森林 + GroupKFold ----
    cv = GroupKFold(n_splits=5)
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    fold_acc, oof = [], np.zeros(len(y), dtype=np.int32)
    for k, (tr, te) in enumerate(cv.split(F, y, groups=g), 1):
        clf.fit(F[tr], y[tr])
        p = clf.predict(F[te])
        oof[te] = p
        a = accuracy_score(y[te], p)
        fold_acc.append(a)
        print(f"    第 {k} 折: 训练 {len(tr):3d} 测试 {len(te):3d}  ->  准确率 {a*100:5.1f}%")
    print()
    acc = accuracy_score(y, oof)
    print(f"[4] 随机森林（GroupKFold 5 折，按轮分组）")
    print(f"    各折: {[f'{a*100:.1f}%' for a in fold_acc]}")
    print(f"    平均: {np.mean(fold_acc)*100:.1f}%  ± {np.std(fold_acc)*100:.1f}%")
    print(f"    整体(合并预测): {acc*100:.1f}%")
    print()

    # ---- 5. 混淆矩阵 ----
    cm = confusion_matrix(y, oof, labels=range(n_cls))
    print("[5] 混淆矩阵（行=真实，列=预测）")
    head = "".join(f"{c[:10]:>12}" for c in classes)
    print(f"{'':<14}{head}")
    for i, c in enumerate(classes):
        row = "".join(f"{v:>12d}" for v in cm[i])
        print(f"{c:<14}{row}")
    print()

    # 每类召回率 = 真正例 / 该类真实总数
    print("    每类召回率:")
    for i, c in enumerate(classes):
        tot = cm[i].sum()
        print(f"      {c:<12} {cm[i,i]:3d}/{tot:3d} = {cm[i,i]/tot*100:5.1f}%")
    print()

    # 最易混的两类
    off = cm.copy()
    np.fill_diagonal(off, 0)
    a, b = np.unravel_index(np.argmax(off), off.shape)
    print(f"    最易混: 真实 {classes[a]} 被判成 {classes[b]}（{off[a,b]} 个窗口）")
    print()

    # ---- 6. 特征重要性 ----
    clf.fit(F, y)   # 用全量重训一次，只为看重要性
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:8]
    print("[6] 特征重要性 Top 8")
    for i in order:
        bar = "#" * int(imp[i] * 100)
        print(f"    {FEATURE_NAMES[i]:<16} {imp[i]*100:5.1f}%  {bar}")
    print()

    # ---- 7. 结论 ----
    print("=" * 70)
    print(f"结论：随机森林 {acc*100:.1f}%  vs  规则基线 {rule_acc*100:.1f}%  （门槛 90%）")
    if acc >= 0.90:
        print("  [OK] 过线。421 个窗口够用，W8 可以导出模型、W9 转 int8。")
    elif acc >= 0.80:
        print("  [!!] 差一点。优先看混淆矩阵里那一对，通常是采集规范问题，不是模型问题。")
    else:
        print("  [X ] 不够。先别加模型复杂度，回 W6-A 查清洗（窗口是否跨轮、剔除阈值）。")
    print("=" * 70)


if __name__ == "__main__":
    main()
