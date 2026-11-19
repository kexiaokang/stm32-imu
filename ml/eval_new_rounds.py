"""
W7-B 补数据评估：补采的 2 轮到底有没有用
=====================================================================
回答两个问题：
  1. 加了 2 轮之后，准确率 95.5% -> 95.1% 是"变差"还是"分母变大"？
     （做法：只在【原来那 421 个窗口】上算准确率，和之前同口径比）
  2. 换了地方采的这 2 轮，模型考得怎么样？（真正意义上的跨环境测试）

不训练新模型，只做评估。
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_baseline import NPZ, extract_features
from clean_data import (
    load_csv, drop_zero_rows, window_span,
    MANIFEST, CLASSES, WIN, STRIDE, STATIC_T,
)

NEW_ROUNDS = 2      # 最后 2 轮是新采的


def build_with_round_id():
    """照 clean_data.py 的流程切窗，但额外记住每个窗口来自【第几轮】。"""
    X, y, rnd = [], [], []
    for label, name in enumerate(CLASSES):
        rounds = MANIFEST[name]
        raw, _ = load_csv(name)
        arr, _ = drop_zero_rows(raw)
        arr = arr[max(0, len(arr) - rounds * 1024):]

        for r in range(rounds):
            seg = arr[r * 1024:(r + 1) * 1024]
            if len(seg) < WIN:
                continue
            for s0 in range(0, len(seg) - WIN + 1, STRIDE):
                win = seg[s0:s0 + WIN]
                if label != 0 and max(window_span(win)) < STATIC_T:
                    continue
                X.append(win.astype(np.int16))
                y.append(label)
                rnd.append(r)
    return np.stack(X), np.array(y, dtype=np.int8), np.array(rnd, dtype=np.int32)


def rf():
    return RandomForestClassifier(n_estimators=300, random_state=42,
                                  n_jobs=-1, class_weight="balanced")


def main():
    X, y, rnd = build_with_round_id()
    F = extract_features(X)
    cls = list(CLASSES)

    print("=" * 70)
    print("数据集构成")
    print("=" * 70)
    for c in range(len(cls)):
        m = y == c
        rs = sorted(set(rnd[m].tolist()))
        print(f"  {cls[c]:<12} {int(m.sum()):>3} 个窗口，来自第 {rs} 轮")
    print(f"  合计 {len(y)} 个窗口")

    # ---------- 问题 1：同口径比较 ----------
    print()
    print("=" * 70)
    print("问题 1：95.1% vs 95.5%，是变差还是分母变大？")
    print("=" * 70)
    old = ~((y == 4) & (rnd >= 5))          # 去掉 rotate 的第 6/7 轮
    print(f"  原来那批窗口（老数据）: {int(old.sum())} 个")
    print(f"  新补的窗口            : {int((~old).sum())} 个")

    oof = np.zeros(len(y), dtype=np.int32)
    g = np.zeros(len(y), dtype=np.int32)    # 用 (类,轮) 当组
    for c in range(len(cls)):
        m = y == c
        for r in set(rnd[m].tolist()):
            g[m & (rnd == r)] = c * 100 + r
    for tr, te in GroupKFold(n_splits=5).split(F, y, groups=g):
        m = rf(); m.fit(F[tr], y[tr]); oof[te] = m.predict(F[te])

    print(f"  全量 {len(y)} 个窗口准确率     : {accuracy_score(y, oof)*100:.1f}%")
    print(f"  只在老窗口上算（同口径）     : {accuracy_score(y[old], oof[old])*100:.1f}%")
    print(f"  只在新补的窗口上算           : {accuracy_score(y[~old], oof[~old])*100:.1f}%  ({int((oof[~old]==y[~old]).sum())}/{int((~old).sum())})")

    # ---------- 问题 2：跨环境测试 ----------
    print()
    print("=" * 70)
    print("问题 2：跨环境测试 —— 老 5 轮训练，只考新 2 轮")
    print("=" * 70)
    te = (y == 4) & (rnd >= 5)
    tr = ~te
    m = rf(); m.fit(F[tr], y[tr])
    p = m.predict(F[te])
    acc = accuracy_score(y[te], p)
    print(f"  训练: {int(tr.sum())} 个窗口（老数据全部）")
    print(f"  测试: {int(te.sum())} 个窗口（新采的 2 轮 rotate）")
    print(f"  准确率(=召回率): {acc*100:.1f}%   ({int((p==y[te]).sum())}/{int(te.sum())})")
    print("  判错的都判成了什么:")
    for c in range(len(cls)):
        n = int((p == c).sum() - (y[te][p == c] == c).sum())
        if n:
            print(f"    -> {cls[c]:<12} {n} 个")
    print(f"  平均置信度: {m.predict_proba(F[te]).max(axis=1).mean()*100:.1f}%")

    # ---------- 对照：老数据当测试集 ----------
    print()
    print("=" * 70)
    print("对照：反过来考老数据（训练时去掉 rotate 第 1~5 轮）")
    print("=" * 70)
    te2 = (y == 4) & (rnd < 5)
    tr2 = ~te2
    m2 = rf(); m2.fit(F[tr2], y[tr2])
    p2 = m2.predict(F[te2])
    print(f"  测试: {int(te2.sum())} 个窗口（老 5 轮 rotate）")
    print(f"  准确率: {accuracy_score(y[te2], p2)*100:.1f}%")
    print(f"  平均置信度: {m2.predict_proba(F[te2]).max(axis=1).mean()*100:.1f}%")

    # ---------- 每轮单独看新数据 ----------
    print()
    print("=" * 70)
    print("新 2 轮分别考得怎么样")
    print("=" * 70)
    for r in range(5, 5 + NEW_ROUNDS):
        mask = (y == 4) & (rnd == r)
        if mask.sum() == 0:
            continue
        print(f"  第 {r+1} 轮: {int(mask.sum()):>2} 个窗口，认对 {int((p[mask[te]]==4).sum()):>2} 个"
              f"  = {accuracy_score(y[mask], p[mask[te]])*100:5.1f}%")

    print()
    print("=" * 70)
    print("结论怎么读")
    print("=" * 70)
    print("  · 同口径老窗口分 >= 95.5%  -> 没变差，95.1% 只是分母变大")
    print("  · 跨环境（新 2 轮）分高    -> 模型能泛化到新环境")
    print("  · 跨环境分低但置信度也低   -> 模型知道自己不确定（可接受）")
    print("  · 跨环境分低但置信度很高   -> 危险：自信地判错，需要加时序特征")
    print("=" * 70)


if __name__ == "__main__":
    main()
