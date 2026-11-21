"""
W9-3：导出"端到端"测试数据（全部 451 个窗口）
=====================================================================
输出：firmware/full_test.h

里面装的是：
  - 451 个窗口的原始采样点（128 点 × 6 轴，int16）  ≈ 2 MB
  - 每个窗口的真实标签（0~4）
  - 每个窗口属于第几折（用它"没见过"的那个模型预测）
  - Python 侧的预测结果（用于逐个对拍）

为什么要 451 个、而不是挑 25 个：
    25 个只能证明"没写错"，451 个才能证明"整条链路在真实数据上成立"。

为什么每折模型只预测"自己没见过的那一批"：
    如果用全量模型预测训练过的窗口，准确率会是 100%，但那是背答案。
    95.1% 这个数字的全部意义就在于：每个窗口都是被"没看过它的模型"判的。

用法（必须先跑过 export_model.py --folds）：
    python export_full_test.py
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_baseline import NPZ, extract_features, recover_groups   # noqa: E402
from export_model import make_clf                                  # noqa: E402

OUT = HERE.parent / "firmware" / "full_test.h"
N_CLS = 5


def main():
    d = np.load(NPZ, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(int)
    F = extract_features(X)
    g = recover_groups(X, y)
    n = len(y)

    # ---- 重算 5 折，和 export_model.py 完全一样的划分 ----
    fold_of = np.zeros(n, dtype=int)
    exp_pred = np.zeros(n, dtype=int)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(F, y, groups=g)):
        m = make_clf().fit(F[tr], y[tr])
        fold_of[te] = k
        exp_pred[te] = m.predict(F[te])

    acc = (exp_pred == y).mean() * 100
    print("451 窗口，OOF 准确率 %.1f%%（%d/%d）" % (acc, int((exp_pred == y).sum()), n))

    lines = []
    A = lines.append
    A("/* 自动生成，不要手改。生成脚本：ml/export_full_test.py */")
    A("#ifndef FULL_TEST_H")
    A("#define FULL_TEST_H")
    A("")
    A('#include "features.h"')
    A("")
    A("#define N_FULL_CASES %d" % n)
    A("")
    A("/* 轴顺序：ax ay az gx gy gz */")
    A("static const int16_t FULL_RAW[N_FULL_CASES][WIN_LEN][N_AXES] = {")
    for w in X:
        A("  {")
        for row in w:
            A("    {%d,%d,%d,%d,%d,%d},"
              % (int(row[0]), int(row[1]), int(row[2]),
                 int(row[3]), int(row[4]), int(row[5])))
        A("  },")
    A("};")
    A("")
    A("static const int FULL_TRUE[N_FULL_CASES] = {")
    for i in range(0, n, 25):
        A("  " + ", ".join(str(int(v)) for v in y[i:i + 25]) + ",")
    A("};")
    A("")
    A("/* 每个窗口该用第几个折模型（0~4）*/")
    A("static const int FULL_FOLD[N_FULL_CASES] = {")
    for i in range(0, n, 25):
        A("  " + ", ".join(str(int(v)) for v in fold_of[i:i + 25]) + ",")
    A("};")
    A("")
    A("/* Python 侧的预测结果，用于逐个对拍 */")
    A("static const int FULL_EXP_PRED[N_FULL_CASES] = {")
    for i in range(0, n, 25):
        A("  " + ", ".join(str(int(v)) for v in exp_pred[i:i + 25]) + ",")
    A("};")
    A("")
    A("#endif  /* FULL_TEST_H */")
    A("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("  -> %s  %.1f KB" % (OUT.name, OUT.stat().st_size / 1024))
    print()
    print("★ 注意：这个文件必须配 model_data.c 里的 5 个折模型。")
    print("  如果 model_data.c 是全量模型版（没有 FOLD0~FOLD4），先跑：")
    print("      python export_model.py --folds")


if __name__ == "__main__":
    main()
