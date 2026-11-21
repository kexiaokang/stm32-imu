/*
 * W9-2 考官（助手写的，不用改）
 * =====================================================================
 * 对 25 个窗口：拿原始 128×6 采样点 -> 调用你的 extract_features
 * -> 和 Python 算出来的 19 个数比。
 *
 * 判据：相对误差 < 1e-6。
 *   （相对误差 = |C 算的 - Python 算的| / (|Python 算的| + 1e-9)
 *     加 1e-9 是防止"Python 那边刚好是 0"时除以 0）
 */

#include <stdio.h>
#include <math.h>

#include "features.h"
#include "features_test.h"

#define REL_TOL 1e-6

int main(void)
{
    int   bad_cases = 0;
    int   bad_vals  = 0;
    double worst    = 0.0;
    int   worst_case = -1, worst_feat = -1;

    for (int k = 0; k < N_FEAT_CASES; k++) {
        float got[N_FEATURES];

        extract_features(RAW[k], got);

        double worst_here = 0.0;
        int    worst_f    = -1;

        for (int i = 0; i < N_FEATURES; i++) {
            double exp = EXP_FEAT[k][i];
            double rel = fabs((double)got[i] - exp) / (fabs(exp) + 1e-9);
            if (rel > REL_TOL) {
                bad_vals++;
                printf("  窗口%3d  %-14s C=%14.6f  Python=%14.6f  相对误差 %.3e\n",
                       FEAT_SRC_INDEX[k], FEAT_NAMES_C[i],
                       (double)got[i], exp, rel);
            }
            if (rel > worst_here) { worst_here = rel; worst_f = i; }
        }

        if (worst_here > REL_TOL) bad_cases++;

        if (worst_here > worst) {
            worst = worst_here;
            worst_case = k;
            worst_feat = worst_f;
        }

        printf("case %2d | 窗口%3d | 最大相对误差 %.3e | %s\n",
               k, FEAT_SRC_INDEX[k], worst_here,
               (worst_here <= REL_TOL) ? "OK" : "FAIL");
    }

    printf("\n----------------------------------------------\n");
    printf("%d / %d 个窗口通过    %d 个数值超标\n",
           N_FEAT_CASES - bad_cases, N_FEAT_CASES, bad_vals);
    if (worst_case >= 0)
        printf("最差: 窗口%3d 的 %s，相对误差 %.3e\n",
               FEAT_SRC_INDEX[worst_case], FEAT_NAMES_C[worst_feat], worst);

    if (bad_cases == 0) {
        printf("[OK] W9-2 通过：C 端 19 维特征与 Python 一致\n");
        return 0;
    }
    printf("[FAIL] 还没对上。看上面第一条超标的行，对照 features.c 里那个统计量。\n");
    return 1;
}
