/*
 * W9-3 考官（助手写的，不用改）
 * =====================================================================
 * 端到端：451 个窗口，从原始采样点一路走到类别。
 *
 *   原始 128×6  →  extract_features()  →  19 维
 *               →  FOLDS[该窗口属于第几折]  →  forest_predict()
 *               →  类别
 *
 * 三件事：
 *   1. 和 Python 的预测逐个比（必须 451/451 一致）
 *   2. 和真实标签比，算准确率（必须 429/451 = 95.1%）
 *   3. 打混淆矩阵和每类召回
 */

#include <stdio.h>
#include <string.h>

#include "model_data.h"
#include "predict.h"
#include "features.h"
#include "full_test.h"

int main(void)
{
    int     agree = 0;                  /* 和 Python 一致的个数 */
    int     right = 0;                  /* 和真实标签相同的个数 */
    int     first_bad = -1;
    int     conf[N_CLASSES][N_CLASSES];

    memset(conf, 0, sizeof(conf));

    for (int k = 0; k < N_FULL_CASES; k++) {
        float f[N_FEATURES];
        extract_features(FULL_RAW[k], f);

        const Forest *m = FOLDS[FULL_FOLD[k]];
        int got = forest_predict(m, f);

        if (got == FULL_EXP_PRED[k]) agree++;
        else if (first_bad < 0)     first_bad = k;

        if (got == FULL_TRUE[k]) right++;
        conf[FULL_TRUE[k]][got]++;
    }

    const char *names[N_CLASSES] = {
        "0_static", "1_pickup", "2_shake_lr", "3_flip", "4_rotate"
    };

    printf("\n================ 结果 ================\n");
    printf("窗口总数                : %d\n", N_FULL_CASES);
    printf("与 Python 预测一致      : %d / %d\n", agree, N_FULL_CASES);
    printf("与真实标签一致（准确率）: %d / %d = %.1f%%\n",
           right, N_FULL_CASES, 100.0 * right / N_FULL_CASES);

    printf("\n混淆矩阵（行=真实，列=预测）\n");
    printf("%-12s", "");
    for (int c = 0; c < N_CLASSES; c++) printf("%12s", names[c]);
    printf("\n");
    for (int r = 0; r < N_CLASSES; r++) {
        printf("%-12s", names[r]);
        for (int c = 0; c < N_CLASSES; c++) printf("%12d", conf[r][c]);
        printf("\n");
    }

    printf("\n每类召回率（这一类的窗口认对了多少）\n");
    for (int r = 0; r < N_CLASSES; r++) {
        int tot = 0, hit = conf[r][r];
        for (int c = 0; c < N_CLASSES; c++) tot += conf[r][c];
        printf("  %-12s %3d / %3d = %5.1f%%\n", names[r], hit, tot,
               tot ? 100.0 * hit / tot : 0.0);
    }

    printf("\n--------------------------------------\n");
    int ok = 1;
    if (agree != N_FULL_CASES) {
        printf("[FAIL] 有 %d 个窗口和 Python 判得不一样，第一个是窗口 %d\n",
               N_FULL_CASES - agree, first_bad);
        ok = 0;
    }
    /* 门槛：≥ 94%（451 × 0.94 = 424）。树数变化会让准确率小幅波动，
       所以这里不硬编码具体数字，只守一条底线。 */
    if (right < 424) {
        printf("[FAIL] 准确率低于 94%%（< 424/451），实际 %d\n", right);
        ok = 0;
    }
    if (ok) {
        printf("[OK] W10-1 通过：C 端与 Python 逐窗口一致，准确率 %.1f%%\n",
               100.0 * right / N_FULL_CASES);
        return 0;
    }
    return 1;
}
