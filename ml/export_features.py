"""
W9-2：导出"特征提取"的测试数据
=====================================================================
输入：ml/dataset.npz
输出：firmware/features_test.h

里面装的是：
  - 25 个窗口的**原始采样点**（128 个时间点 × 6 轴，int16）
  - 这 25 个窗口的**正确 19 维特征**（由 train_baseline.extract_features 算出来）

C 端要做的：拿同样 128×6 的原始数据，自己算出 19 个数，和这里的对照。
误差小于 1e-6（相对值）就算通过。

为什么用"原始采样点"而不是直接给特征：
    因为 W9-2 要验证的就是"从原始点到 19 个特征"这一段。
    给特征就变成 W9-1 了，什么都验证不了。

用法：
    python export_features.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_baseline import NPZ, extract_features, FEATURE_NAMES   # noqa: E402
from export_model import pick_cases                               # noqa: E402

OUT = HERE.parent / "firmware" / "features_test.h"
OUT_BOARD = HERE.parent / "firmware" / "board" / "features_test.h"   # 上板版：只有 5 个窗口


def dnum(x):
    """double 字面量。17 位有效数字保证 float64 完全还原。"""
    s = "%.17g" % float(x)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", action="store_true",
                    help="上板版：只导 5 个窗口（每类 1 个）到 firmware/board/，"
                         "Flash 只有 512 KB，25 个窗口塞不下")
    args = ap.parse_args()

    d = np.load(NPZ, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(int)

    idx = pick_cases(y)                 # 和 W9-1 用同一批 25 个窗口（每类 5 个，按类别排好）
    if args.board:
        # pick_cases 的顺序是 0,0,0,0,0, 1,1,1,1,1, ...，取下标 0/5/10/15/20 就是每类 1 个
        idx = [idx[0], idx[5], idx[10], idx[15], idx[20]]
        out = OUT_BOARD
    else:
        out = OUT

    raw = X[idx]                        # (N, 128, 6) int16
    feat = extract_features(X[idx])     # (N, 19) float64  ← 标准答案
    true_label = y[idx]                 # 真实类别，报错时和预测类别对照

    lines = []
    lines.append("/* 自动生成，不要手改。生成脚本：ml/export_features.py */")
    lines.append("#ifndef FEATURES_TEST_H")
    lines.append("#define FEATURES_TEST_H")
    lines.append("")
    lines.append('#include "features.h"')
    lines.append("")
    lines.append("#define N_FEAT_CASES %d" % len(idx))
    lines.append("")
    lines.append("/* 每个用例：128 个时间点 × 6 轴，原始 int16 读数 */")
    lines.append("/* 轴顺序：ax ay az gx gy gz（和 CSV、和工程里完全一致）*/")
    lines.append("static const int16_t RAW[N_FEAT_CASES][WIN_LEN][N_AXES] = {")
    for w in raw:
        lines.append("  {")
        for row in w:
            lines.append("    {%d,%d,%d,%d,%d,%d},"
                         % (int(row[0]), int(row[1]), int(row[2]),
                            int(row[3]), int(row[4]), int(row[5])))
        lines.append("  },")
    lines.append("};")
    lines.append("")
    lines.append("/* 标准答案：19 维特征（由 sklearn 训练脚本里的 extract_features 算出）*/")
    lines.append("static const double EXP_FEAT[N_FEAT_CASES][N_FEATURES] = {")
    for row in feat:
        lines.append("  {%s}," % ", ".join(dnum(v) for v in row))
    lines.append("};")
    lines.append("")
    lines.append("/* 19 维特征的名字，报错时好定位 */")
    lines.append("static const char *FEAT_NAMES_C[N_FEATURES] = {")
    for i, nm in enumerate(FEATURE_NAMES):
        lines.append('  "%s",%s' % (nm, "  /* %d */" % i))
    lines.append("};")
    lines.append("")
    lines.append("/* 每个用例来自数据集第几个窗口 */")
    lines.append("static const int FEAT_SRC_INDEX[N_FEAT_CASES] = {")
    lines.append("  " + ", ".join(str(int(i)) for i in idx))
    lines.append("};")
    lines.append("")
    lines.append("/* 真实类别（不是模型预测），用来判断类别这一步对不对 */")
    lines.append("static const int FEAT_TRUE_LABEL[N_FEAT_CASES] = {")
    lines.append("  " + ", ".join(str(int(v)) for v in true_label))
    lines.append("};")
    lines.append("")
    lines.append("#endif  /* FEATURES_TEST_H */")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("窗口 %d 个，每个 %d 点 × %d 轴%s"
          % (len(idx), raw.shape[1], raw.shape[2], "（上板版）" if args.board else ""))
    print("  -> %s  %.1f KB" % (out, out.stat().st_size / 1024))
    print()
    print("特征范围（用来判断算错没算错）：")
    for i, nm in enumerate(FEATURE_NAMES):
        col = feat[:, i]
        print("  %-14s %12.3f ~ %12.3f" % (nm, col.min(), col.max()))


if __name__ == "__main__":
    main()
