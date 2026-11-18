"""
W7-A 选型实测：为什么不选别的
=====================================================================
回答一个问题——**同一份数据（421 窗口 × 19 特征，5 类），
换不同算法，准确率差多少？**

这样做的好处：以后拿到新项目，"该用什么模型"不用猜，
照着这张表的方法论做一遍，半小时就有答案。

统一条件（公平比较的前提）：
  · 同一份特征（19 维，见 train_baseline.py）
  · 同一套交叉验证（GroupKFold 5 折，按轮分组）
  · 同一个 random_state

最后额外跑一次"错误的切法"（随机 StratifiedKFold），
用来量化"泄露会让准确率虚高多少"——这个数以前是凭经验说的，
现在实测。
"""

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_baseline import (
    NPZ, recover_groups, extract_features, rule_predict,
)

SEED = 42
N_SPLIT = 5


def build_models():
    """候选模型清单。顺序 = 从最笨到最复杂（这也是实际该试的顺序）。"""
    return [
        # 0. 全猜最多的那一类 —— 任何模型都必须打败它，否则等于没学
        ("永远猜最多类", DummyClassifier(strategy="most_frequent")),

        # 1. 线性模型：假设"一条直线能分开"
        ("逻辑回归(标准化)", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=SEED),
        )),

        # 2. 最近的邻居：谁离我最近，我就是谁
        ("KNN k=5(标准化)", make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5),
        )),

        # 3. 单棵决策树：一串 if-else
        ("单棵决策树", DecisionTreeClassifier(random_state=SEED)),

        # 4. 随机森林：300 棵树投票（W7-A 用的就是这个）
        ("随机森林 300 棵", RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1,
            class_weight="balanced",
        )),

        # 5. 梯度提升：一棵接一棵，每棵纠正前一棵的错
        ("梯度提升 HistGB", HistGradientBoostingClassifier(
            random_state=SEED, max_iter=200,
        )),

        # 6. 小型神经网络：421 个样本够不够，看结果
        ("神经网络 MLP(32)", MLPClassifier(
            hidden_layer_sizes=(32,), max_iter=3000,
            random_state=SEED,
        )),
    ]


def run_cv(model, F, y, groups):
    """跑 5 折，返回 (准确率, macro-F1, 耗时秒)。"""
    cv = GroupKFold(n_splits=N_SPLIT)
    oof = np.zeros(len(y), dtype=np.int32)
    t0 = time.time()
    for tr, te in cv.split(F, y, groups=groups):
        model.fit(F[tr], y[tr])
        oof[te] = model.predict(F[te])
    dt = time.time() - t0
    return accuracy_score(y, oof), f1_score(y, oof, average="macro"), dt


def main():
    d = np.load(NPZ, allow_pickle=True)
    X, y = d["X"], d["y"].astype(np.int32)
    classes = [str(c) for c in d["classes"]]

    g = recover_groups(X, y)
    F = extract_features(X)

    print("=" * 78)
    print("W7-A 选型实测：同一份数据，换算法")
    print("=" * 78)
    print(f"样本 {len(y)} 个窗口 × {F.shape[1]} 维特征，{len(classes)} 类")
    print(f"交叉验证：GroupKFold {N_SPLIT} 折，按「轮」分组（{int(g.max())+1} 组）")
    print(f"指标：准确率 / macro-F1（每类同等重要）")
    print()

    # ---- 基线1：手写规则（0 训练）----
    rule_acc = accuracy_score(y, rule_predict(X))
    rule_f1 = f1_score(y, rule_predict(X), average="macro")
    rows = [("手写阈值规则(0训练)", rule_acc, rule_f1, 0.0)]   # 固定放第一行

    # ---- 候选模型 ----
    for name, model in build_models():
        acc, f1, dt = run_cv(model, F, y, g)
        rows.append((name, acc, f1, dt))
        print(f"  {name:<20} 准确率 {acc*100:5.1f}%   F1 {f1:5.3f}   {dt:5.2f}s")

    print()
    print("-" * 78)
    print(f"{'方法':<22}{'准确率':>9}{'macro-F1':>11}{'训练耗时':>10}")
    print("-" * 78)
    for name, acc, f1, dt in rows:
        print(f"{name:<22}{acc*100:>8.1f}%{f1:>11.3f}{dt:>9.2f}s")
    print("-" * 78)

    # 最好的是谁
    best = max(rows, key=lambda r: (r[1], r[2]))
    print(f"最好: {best[0]}  {best[1]*100:.1f}%")

    # ---- 关键实验：错误切法会让准确率虚高多少 ----
    print()
    print("=" * 78)
    print("对照实验：把 GroupKFold 换成随机 StratifiedKFold")
    print("=" * 78)
    print("含义：随机切分时，同一轮里 50% 重叠的相邻窗口会同时落进训练集和测试集，")
    print("      模型等于考前看过卷子。这个差就是「泄露量」。")
    print()
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                n_jobs=-1, class_weight="balanced")
    mean_g, mean_r = [], []
    for seed in range(5):                     # 随机切分有运气成分，跑 5 次取平均
        cv_g = GroupKFold(n_splits=N_SPLIT)
        oof_g = np.zeros(len(y), dtype=np.int32)
        for tr, te in cv_g.split(F, y, groups=g):
            rf.fit(F[tr], y[tr])
            oof_g[te] = rf.predict(F[te])
        mean_g.append(accuracy_score(y, oof_g))

        cv_r = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=seed)
        oof_r = np.zeros(len(y), dtype=np.int32)
        for tr, te in cv_r.split(F, y):
            rf.fit(F[tr], y[tr])
            oof_r[te] = rf.predict(F[te])
        mean_r.append(accuracy_score(y, oof_r))

    mg, mr = np.mean(mean_g) * 100, np.mean(mean_r) * 100
    print(f"  按轮分组 GroupKFold : {mg:5.2f}%")
    print(f"  随机切分 Stratified : {mr:5.2f}%")
    print(f"  → 泄露导致的虚高    : {mr - mg:+5.2f} 个百分点")
    print()
    print("  注意：虚高不多 ≠ 可以随便切。421 个样本里真正独立的只有 33 组，")
    print("       随机切分下模型仍能靠记忆拿到一部分分，只是这批数据「太好分」，")
    print("       ceil 也就 100%，所以差距被压小了。WRONG 的切法一旦用在难数据上，")
    print("       虚高会到 20 个点以上。")
    print("=" * 78)


if __name__ == "__main__":
    main()
