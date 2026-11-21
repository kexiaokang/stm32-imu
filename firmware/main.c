/*
 * W9-1 考官（助手写的，你不用改）
 * =====================================================================
 * 它做三件事：
 *   1. 对 25 个测试用例，调用你的 forest_predict，看类别对不对
 *   2. 自己也把 300 棵树累加一遍，和 sklearn 的 predict_proba 逐个比
 *      （这样即使类别蒙对了，概率不对也会被抓出来）
 *   3. 打印通过数
 */

#include <stdio.h>
#include <math.h>

#include "model_data.h"
#include "predict.h"
#include "test_cases.h"

#define TOL 1e-3f          /* 概率容差：float 累加 300 棵树，1e-3 已经很严 */

int main(void)
{
    int   ok    = 0;
    float worst = 0.0f;

    for (int i = 0; i < N_CASES; i++) {
        const float *x = CASE_X[i];

        /* ---- 概率对拍：自己累加 300 棵树，再除以棵数 ---- */
        float acc[N_CLASSES] = {0.0f};
        for (int t = 0; t < FOREST_FULL.n_trees; t++)
            tree_predict(&FOREST_FULL.trees[t], x, acc);
        for (int c = 0; c < N_CLASSES; c++)
            acc[c] /= (float)FOREST_FULL.n_trees;

        float diff = 0.0f;
        for (int c = 0; c < N_CLASSES; c++) {
            float d = fabsf(acc[c] - CASE_PROB[i][c]);
            if (d > diff) diff = d;
        }
        if (diff > worst) worst = diff;

        /* ---- 类别对拍 ---- */
        int got  = forest_predict(&FOREST_FULL, x);
        int pass = (got == CASE_EXPECT[i]) && (diff <= TOL);
        ok += pass;

        printf("case %2d | 窗口%3d | 期望 %d | 得到 %2d | 概率误差 %.6f | %s\n",
               i, CASE_SRC_INDEX[i], CASE_EXPECT[i], got, diff,
               pass ? "OK" : "FAIL");
    }

    printf("\n----------------------------------------------\n");
    printf("%d / %d 用例通过    最大概率误差 %.6f（容差 %.0e）\n",
           ok, N_CASES, worst, (double)TOL);

    if (ok == N_CASES && worst <= TOL) {
        printf("[OK] W9-1 通过：C 端的树遍历和 sklearn 完全一致\n");
        return 0;
    }
    printf("[FAIL] 还没对上。从第一个 FAIL 的用例开始查。\n");
    return 1;
}
