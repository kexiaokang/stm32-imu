"""
W9：把训练好的随机森林导出成 C 数组
=====================================================================
输入：ml/dataset.npz
输出：firmware/model_data.h   —— 数据结构定义（Node / Tree / Forest）
      firmware/model_data.c   —— 300 棵树的节点数组（约 1 MB 源码，别去读它）
      firmware/test_cases.h   —— 25 个测试窗口：19 维特征 + 期望类别 + 期望概率

为什么要导出，而不在板子上用 Python：
    板子上没有 Python，也没有足够的算力跑 sklearn。
    但随机森林推理只需要"比较 + 查表"，一个 while 循环就够。

为什么导出【节点数组】而不是 if-else：
    300 棵树写成 if-else 会有几万行嵌套，编译都过不去，也没法用循环跑。
    节点数组 = 把树拍平成一张表，用一个下标在表里走。

⚠️ 一个必须知道的坑：
    拿全部 451 个窗口训练、再用这 451 个自测，准确率是 100%——
    那是【训练集准确率】，不是泛化能力。
    要复现 95.1%，必须导出 5 个折模型（--folds），
    每个窗口只用"没见过它的那一折"预测。

用法：
    python export_model.py            # 只导出全量模型（W9-1 用这个）
    python export_model.py --folds    # 额外导出 5 个折模型（W9-3 用，文件大 5 倍）
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_baseline import NPZ, extract_features, recover_groups   # noqa: E402

OUT_DIR = HERE.parent / "firmware"
OUT_DIR_BOARD = HERE.parent / "firmware" / "board"   # --board：只含全量模型，上板用
N_EST = 100          # W10-1：从 300 砍到 100（体积降到 1/3，实测准确率不降反升）
SEED = 42
N_CASES = 25          # 导出多少个测试用例


# ---------------------------------------------------------------- 训练
def make_clf():
    return RandomForestClassifier(n_estimators=N_EST, random_state=SEED,
                                  n_jobs=-1, class_weight="balanced")


def leaf_prob(tree, node):
    """一个叶子的 5 类概率：sv 是各类计数，归一化后才是概率。

    为什么不能用计数直接比大小：不同叶子的样本数不同
    （有的叶子 1 个样本，有的 30 个），不归一化就等于给"大叶子"加权。
    sklearn 的 predict_proba 是归一化后再平均的。
    """
    v = tree.value[node, 0, :].astype(np.float64)
    s = v.sum()
    return v / s if s > 0 else np.zeros_like(v)


# ---------------------------------------------------------------- 写 C
def fnum(x):
    """float 字面量。9 位有效数字保证 float32 来回转换不失真。

    ⚠️ 坑：C 里 `1f` 不合法，必须是 `1.0f`。
    %.9g 对整数会输出 "1"，所以这里要补上小数点。
    """
    s = "%.9g" % float(x)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s + "f"


def emit_model_c(prefix, sym, clf):
    """把一片森林写成 C 源码。prefix 是数组名前缀，sym 是导出的变量名。"""
    trees = clf.estimators_
    out = []
    out.append("/* 自动生成，不要手改。生成脚本：ml/export_model.py */")
    out.append('#include "model_data.h"')
    out.append("")

    for ti, est in enumerate(trees):
        t = est.tree_
        n = int(t.node_count)
        out.append("static const Node %s_t%d[%d] = {" % (prefix, ti, n))
        for i in range(n):
            is_leaf = t.children_left[i] == -1
            if is_leaf:
                p = leaf_prob(t, i)
                ptxt = ", ".join(fnum(v) for v in p)
                out.append("  {0, -1, -1, 0.0f, {%s}}," % ptxt)
            else:
                out.append("  {%d, %d, %d, %s, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},"
                           % (int(t.feature[i]), int(t.children_left[i]),
                              int(t.children_right[i]), fnum(t.threshold[i])))
        out.append("};")
        out.append("")

    out.append("static const Tree %s_trees[%d] = {" % (prefix, len(trees)))
    for ti, est in enumerate(trees):
        out.append("  {%s_t%d, %d, 0}," % (prefix, ti, int(est.tree_.node_count)))
    out.append("};")
    out.append("")
    out.append("const Forest %s = {%s_trees, %d};" % (sym, prefix, len(trees)))
    out.append("")
    return "\n".join(out)


def emit_test_cases(X_test_feat, pred_cls, pred_prob, y_true, idx):
    """25 个测试用例：特征 + 期望类别 + 期望概率（来自 sklearn）。"""
    out = []
    out.append("/* 自动生成，不要手改。生成脚本：ml/export_model.py */")
    out.append("#ifndef TEST_CASES_H")
    out.append("#define TEST_CASES_H")
    out.append("")
    out.append('#include "model_data.h"')
    out.append("")
    out.append("#define N_CASES %d" % len(idx))
    out.append("")
    out.append("/* 每个用例：19 维特征（float）、期望类别、期望的 5 类概率 */")
    out.append("static const float CASE_X[N_CASES][N_FEATURES] = {")
    for row in X_test_feat:
        out.append("  {%s}," % ", ".join(fnum(v) for v in row))
    out.append("};")
    out.append("")
    out.append("static const int CASE_EXPECT[N_CASES] = {")
    out.append("  " + ", ".join(str(int(c)) for c in pred_cls))
    out.append("};")
    out.append("")
    out.append("static const float CASE_PROB[N_CASES][N_CLASSES] = {")
    for row in pred_prob:
        out.append("  {%s}," % ", ".join(fnum(v) for v in row))
    out.append("};")
    out.append("")
    out.append("/* 每个用例来自数据集里的第几个窗口、真实标签是什么 */")
    out.append("static const int CASE_SRC_INDEX[N_CASES] = {")
    out.append("  " + ", ".join(str(int(i)) for i in idx))
    out.append("};")
    out.append("static const int CASE_TRUE_LABEL[N_CASES] = {")
    out.append("  " + ", ".join(str(int(v)) for v in y_true))
    out.append("};")
    out.append("")
    out.append("#endif  /* TEST_CASES_H */")
    out.append("")
    return "\n".join(out)


def pick_cases(y, seed=0):
    """每类挑几个，保证 5 个类别都覆盖到。"""
    rng = np.random.RandomState(seed)
    n_cls = len(np.unique(y))
    per = N_CASES // n_cls
    idx = []
    for c in range(n_cls):
        pool = np.where(y == c)[0]
        idx += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    idx += list(rng.choice(np.setdiff1d(np.arange(len(y)), idx),
                           size=N_CASES - len(idx), replace=False))
    return np.sort(np.array(idx, dtype=int))


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", action="store_true",
                    help="额外导出 5 个折模型（复现 OOF 准确率用，源码会大 5 倍）")
    ap.add_argument("--board", action="store_true",
                    help="只导出【全量模型】到 firmware/board/（上板用，不含 5 折，体积最小）")
    args = ap.parse_args()
    if args.board and args.folds:
        ap.error("--board 和 --folds 不能同时用：上板只要 1 个全量模型")

    d = np.load(NPZ, allow_pickle=True)
    X = d["X"]
    y = d["y"].astype(int)
    class_names = [str(c) for c in d["classes"]]
    F = extract_features(X)
    g = recover_groups(X, y)

    print("=" * 62)
    print("W9  导出随机森林 -> C")
    print("=" * 62)
    print("窗口 %d  特征 %d 维  类别 %s" % (len(y), F.shape[1], class_names))

    # ---- 全量模型 ----
    full = make_clf().fit(F, y)
    n_nodes_full = sum(int(e.tree_.node_count) for e in full.estimators_)
    print()
    print("[1] 全量模型：%d 棵树，%d 个节点，约 %.0f KB"
          % (N_EST, n_nodes_full, n_nodes_full * 32 / 1024))
    print("    拿这 451 个窗口自测 = %.1f%%（训练集准确率，不是泛化能力）"
          % (100 * (full.predict(F) == y).mean()))

    # ---- 5 折（可选）----
    fold_models, fold_of = [], np.zeros(len(y), dtype=int)
    if args.folds:
        print()
        print("[2] 5 折模型（用于复现 OOF 准确率）")
        oof = np.zeros(len(y), dtype=int)
        for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(F, y, groups=g)):
            m = make_clf().fit(F[tr], y[tr])
            fold_models.append(m)
            fold_of[te] = k
            oof[te] = m.predict(F[te])
            print("    第 %d 折: 训练 %3d 测试 %3d 节点 %5d"
                  % (k + 1, len(tr), len(te),
                     sum(int(e.tree_.node_count) for e in m.estimators_)))
        print("    OOF 准确率 %.1f%%（%d/%d）  ← 这就是 95.1%% 的出处"
              % (100 * (oof == y).mean(), int((oof == y).sum()), len(y)))

    # ---- 测试用例 ----
    idx = pick_cases(y)
    xs = F[idx].astype(np.float32)
    pcls = full.predict(F[idx])
    pprob = full.predict_proba(F[idx])
    print()
    print("[3] 测试用例 %d 个（每类 5 个）" % len(idx))

    # ---- 写文件 ----
    out_dir = OUT_DIR_BOARD if args.board else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = (out_dir / "model_data.h")
    src = (out_dir / "model_data.c")
    cases = (out_dir / "test_cases.h")

    tpl_h = HEADER_H_BOARD if args.board else HEADER_H
    hdr.write_text(tpl_h.replace("__N_EST__", str(N_EST)), encoding="utf-8")

    parts = [emit_model_c("full", "FOREST_FULL", full)]
    if fold_models:
        parts.append("/* ===== 5 个折模型 ===== */")
        parts.append("const int FOLD_OF[%d] = {" % len(fold_of))
        parts.append("  " + ", ".join(str(int(v)) for v in fold_of))
        parts.append("};")
        parts.append("const int FOLD_OF_LEN = %d;" % len(fold_of))
        parts.append("")
        for k, m in enumerate(fold_models):
            parts.append(emit_model_c("fold%d" % k, "FOLD%d" % k, m))
        parts.append("const Forest *FOLDS[5] = {&FOLD0, &FOLD1, &FOLD2, &FOLD3, &FOLD4};")
        parts.append("")
    body = HEADER_C_HEAD + "\n".join(parts) + ("" if args.board else HEADER_C_TAIL)
    src.write_text(body, encoding="utf-8")

    cases.write_text(emit_test_cases(xs, pcls, pprob, y[idx], idx), encoding="utf-8")

    print()
    for p in (hdr, src, cases):
        print("  -> %-34s %8.1f KB" % (p.name, p.stat().st_size / 1024))
    print()
    print("完成。下一步：写 firmware/predict.c 里的两个函数。")


# ---------------------------------------------------------------- 固定模板
HEADER_H = """\
/* 自动生成，不要手改。生成脚本：ml/export_model.py */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#define N_CLASSES   5
#define N_FEATURES  19

/*
 * 一棵树被"拍平"成一张节点表。
 *
 *   内部节点：left / right >= 0，prob 全 0
 *   叶子节点：left == -1
 *
 * 走法（就这一个循环，整片森林都是它）：
 *
 *   int node = 0;
 *   while (t->nodes[node].left != -1) {
 *       node = (x[t->nodes[node].feature] <= t->nodes[node].threshold)
 *              ? t->nodes[node].left
 *              : t->nodes[node].right;
 *   }
 *   // 到这里 node 是叶子，概率在 t->nodes[node].prob[0..4]
 */
typedef struct {
    int16_t feature;          /* 用第几维特征比较（0~18） */
    int16_t left;             /* 左孩子下标；-1 表示叶子 */
    int16_t right;            /* 右孩子下标；-1 表示叶子 */
    float   threshold;        /* x[feature] <= threshold → 走左边 */
    float   prob[N_CLASSES];  /* 叶子：归一化后的 5 类概率；内部节点：全 0 */
} Node;

typedef struct {
    const Node *nodes;
    int         n_nodes;
    int         root;         /* 根节点下标，目前都是 0 */
} Tree;

typedef struct {
    const Tree *trees;
    int         n_trees;
} Forest;

extern const Forest FOREST_FULL;   /* __N_EST__ 棵树，用全部 451 个窗口训练 */

extern const char *CLASS_NAMES[N_CLASSES];

/*
 * 5 个折模型（只有 `python export_model.py --folds` 才会生成定义）。
 * FOLDS[k] 用于预测"第 k 折的测试窗口"，即 FULL_FOLD[i] == k 的那些窗口。
 */
extern const Forest *FOLDS[5];
extern const int     FOLD_OF[];
extern const int     FOLD_OF_LEN;

#endif  /* MODEL_DATA_H */
"""

# 上板版：没有 5 折模型的声明（省体积，也避免链接到不存在的符号）
HEADER_H_BOARD = """\
/* 自动生成，不要手改。生成脚本：ml/export_model.py --board */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#define N_CLASSES   5
#define N_FEATURES  19

/*
 * 一棵树被"拍平"成一张节点表。
 *
 *   内部节点：left / right >= 0，prob 全 0
 *   叶子节点：left == -1
 *
 * 走法（就这一个循环，整片森林都是它）：
 *
 *   int node = 0;
 *   while (t->nodes[node].left != -1) {
 *       node = (x[t->nodes[node].feature] <= t->nodes[node].threshold)
 *              ? t->nodes[node].left
 *              : t->nodes[node].right;
 *   }
 *   // 到这里 node 是叶子，概率在 t->nodes[node].prob[0..4]
 */
typedef struct {
    int16_t feature;          /* 用第几维特征比较（0~18） */
    int16_t left;             /* 左孩子下标；-1 表示叶子 */
    int16_t right;            /* 右孩子下标；-1 表示叶子 */
    float   threshold;        /* x[feature] <= threshold → 走左边 */
    float   prob[N_CLASSES];  /* 叶子：归一化后的 5 类概率；内部节点：全 0 */
} Node;

typedef struct {
    const Node *nodes;
    int         n_nodes;
    int         root;         /* 根节点下标，目前都是 0 */
} Tree;

typedef struct {
    const Tree *trees;
    int         n_trees;
} Forest;

extern const Forest FOREST_FULL;   /* __N_EST__ 棵树，用全部 451 个窗口训练 */

extern const char *CLASS_NAMES[N_CLASSES];

#endif  /* MODEL_DATA_H */
"""

HEADER_C_HEAD = """\
/* 自动生成，不要手改。生成脚本：ml/export_model.py */
#include "model_data.h"

const char *CLASS_NAMES[N_CLASSES] = {
    "0_static", "1_pickup", "2_shake_lr", "3_flip", "4_rotate"
};

"""

HEADER_C_TAIL = """\
/* 折模型相关（只有 export_model.py --folds 才会生成） */
extern const Forest *FOLDS[5];
extern const int     FOLD_OF[];
extern const int     FOLD_OF_LEN;
"""


if __name__ == "__main__":
    main()
