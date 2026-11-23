/*
 * W10-2 第二步：在板子上验证"特征提取"
 * =====================================================================
 *
 * 这一步仍然【不接传感器】。喂的是 5 组已经录好的原始采样
 * （每类动作挑 1 个窗口，正好覆盖 5 类）：
 *
 *   128 个时间点 x 6 轴 = 768 个 int16  ->  features.c  ->  19 维特征
 *
 * 为什么第一步先验证树、第二步才验证特征：
 *   第一步已经证明"给 19 维特征，板子能算对类别"。
 *   现在把数据来源从"写死的特征表"换成"从原始点现算特征"。
 *   万一这一步错了，会同时看到"特征不对"和"类别不对"；
 *   但因为树已经单独验证过，就能确定毛病出在 features.c 这一层。
 *
 * 为什么只放 5 个窗口（不是 25 个）：
 *   板子 Flash 只有 512 KB，当前固件已经占 486 KB（95%）。
 *   25 个窗口要多吃 43 KB，直接编译失败（L6406E）。
 *   5 个窗口只要 8.6 KB，够验证算法了：5 类各一个，19 维全比。
 *
 * 判据：
 *   ① 19 维特征与 Python 的相对误差 <= 1000 ppb（即 1e-6）  <- 主要判据
 *   ② 顺带报一下"预测类别 vs 真实类别"，只作参考
 *      （模型本身有 4% 的错误率，类别错 1 个不代表特征算错了）
 *
 * 怎么用：
 *   烧录后，串口里敲：  feat_test
 */

#include <rtthread.h>

#include "model_data.h"     /* N_FEATURES / N_CLASSES */
#include "predict.h"        /* forest_predict */
#include "features.h"       /* extract_features / WIN_LEN / N_AXES */
#include "features_test.h"  /* RAW[5][128][6] / EXP_FEAT[5][19] / FEAT_TRUE_LABEL[5] */

#define TOL_PPB   1000L     /* 相对误差上限：1000 ppb = 1e-6 */

/* 自己写一个 double 绝对值，避免为了 fabs() 去链接浮点数学库 */
static double dabs(double v)
{
    return (v < 0.0) ? -v : v;
}

static int feat_test(void)
{
    int  ok        = 0;     /* 特征在容差内的窗口数 */
    int  cls_ok    = 0;     /* 预测类别 == 真实类别的窗口数 */
    long worst_ppb = 0;     /* 最差的那个误差 */
    int  worst_case = -1;
    int  worst_dim  = -1;

    rt_kprintf("\r\n");
    rt_kprintf("===== W10-2 第二步：%d 组 128x6 原始采样 =====\r\n", N_FEAT_CASES);
    rt_kprintf("验证 features.c：原始点 -> 19 维特征 -> 类别\r\n\r\n");

    for (int k = 0; k < N_FEAT_CASES; k++)
    {
        float f[N_FEATURES];

        /* ① 板子上现算 19 维特征（这一行就是被测对象） */
        extract_features(RAW[k], f);

        /* ② 和 Python 的标准答案逐维对比，取最差的一维 */
        long worst = 0;
        int  dim   = 0;

        for (int i = 0; i < N_FEATURES; i++)
        {
            double ref  = EXP_FEAT[k][i];
            double base = dabs(ref);
            if (base < 1.0)
                base = 1.0;     /* 防止 ref 接近 0 时除出天文数字 */

            /* 相对误差换算成 ppb（十亿分之一），用整数打印，避开串口不支持 %f 的坑 */
            long ppb = (long)(dabs((double)f[i] - ref) / base * 1e9 + 0.5);

            if (ppb > worst)
            {
                worst = ppb;
                dim   = i;
            }
        }

        if (worst > worst_ppb)
        {
            worst_ppb  = worst;
            worst_case = k;
            worst_dim  = dim;
        }

        /* ③ 再走一遍森林，验证"特征 -> 类别"这一段没被破坏 */
        int got  = forest_predict(&FOREST_FULL, f);
        int true = FEAT_TRUE_LABEL[k];

        int feat_ok = (worst <= TOL_PPB);

        if (feat_ok)
            ok++;
        if (got == true)
            cls_ok++;

        rt_kprintf("case %d | 窗口%3d | 误差%6ldppb(%-12s) | 类别 真实%d/预测%d | %s\r\n",
                   k, FEAT_SRC_INDEX[k], worst, FEAT_NAMES_C[dim],
                   true, got, feat_ok ? "OK" : "FAIL-特征");
    }

    rt_kprintf("\r\n----------------------------------------\r\n");
    rt_kprintf("特征 %d / %d 一致（容差 %ld ppb）\r\n", ok, N_FEAT_CASES, TOL_PPB);
    rt_kprintf("类别 %d / %d 判对（参考项，模型本身约 4%% 出错率）\r\n", cls_ok, N_FEAT_CASES);
    rt_kprintf("最大特征误差：%ld ppb", worst_ppb);
    if (worst_case >= 0)
        rt_kprintf("（case %d 的 %s）", worst_case, FEAT_NAMES_C[worst_dim]);
    rt_kprintf("\r\n");

    if (ok == N_FEAT_CASES)
        rt_kprintf("[OK] W10-2 第二步通过：板子上的特征提取与 PC 完全一致\r\n");
    else
        rt_kprintf("[FAIL] 特征不一致，查 features.c：滑窗长度、标准差除 N、维度顺序\r\n");

    return 0;
}

MSH_CMD_EXPORT(feat_test, W10-2 feature extract test);
