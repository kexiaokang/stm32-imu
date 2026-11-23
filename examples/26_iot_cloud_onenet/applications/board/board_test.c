/*
 * W10-2 第一步：在板子上验证"树遍历"
 * =====================================================================
 *
 * 这一步【不接传感器、不算特征】。
 * 直接喂 25 组已经算好的 19 维特征，看板子算出的类别和 PC 上是否一致。
 *
 * 为什么要先这样：
 *   如果一上来就接传感器，出错时你分不清是"树走错了"还是"数据采错了"。
 *   先把树单独验证对，再接数据 —— 和 W9 的三步是同一个套路。
 *
 * 怎么用：
 *   烧录后，串口里敲：  board_test
 *   （和 icm_demo、i2c_scan 一样，都是 MSH 命令）
 *
 * 判据：
 *   25 / 25 通过。任何一个 FAIL 都说明板子上的树遍历和 PC 不一致。
 */

#include <rtthread.h>

#include "model_data.h"
#include "predict.h"
#include "test_cases.h"

static int board_test(void)
{
    int ok = 0;

    rt_kprintf("\r\n");
    rt_kprintf("===== W10-2 第一步：喂 25 组固定特征 =====\r\n");
    rt_kprintf("每类 5 个，不接传感器\r\n\r\n");

    for (int k = 0; k < N_CASES; k++)
    {
        /* 只用 19 维特征走森林，返回 0~4 */
        int got = forest_predict(&FOREST_FULL, CASE_X[k]);

        int exp  = CASE_EXPECT[k];       /* PC 上 sklearn 的预测 */
        int pass = (got == exp);

        if (pass)
            ok++;

        rt_kprintf("case %2d | 窗口%3d | 真实=%d 期望=%d 得到=%d | %s\r\n",
                   k, CASE_SRC_INDEX[k], CASE_TRUE_LABEL[k],
                   exp, got, pass ? "OK" : "FAIL");
    }

    rt_kprintf("\r\n----------------------------------------\r\n");
    rt_kprintf("%d / %d 通过\r\n", ok, N_CASES);

    if (ok == N_CASES)
        rt_kprintf("[OK] W10-2 第一步通过：板子上的树遍历与 PC 完全一致\r\n");
    else
        rt_kprintf("[FAIL] 有 %d 个不一致，先查 predict.c 的比较符和叶子判断\r\n",
                   N_CASES - ok);

    return 0;
}

MSH_CMD_EXPORT(board_test, W10-2 board test);
