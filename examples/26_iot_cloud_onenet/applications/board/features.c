/*
 * W9-2：19 维特征提取（C 版）
 * =====================================================================
 * 这个文件是 Python 里 extract_features() 的逐行翻译。
 *
 * 唯一的要求：结果必须和 Python 一模一样。
 * 所以每一个算法细节都照抄，不许"顺手优化"：
 *   - 滑动平均是"尾部对齐"，前 9 个点用"已有部分的平均"
 *   - 标准差除以 N（不是 N-1），和 numpy 的默认一致
 *   - az 负值比例的判据是 "严格小于 0"，不含 0
 *
 * 为什么中间用 double 算、最后才存成 float：
 *   板上是单精度浮点（M4F 的 FPU 只支持 float），
 *   但"求和 128 个数"这种累加用 float 会掉精度。
 *   所以先用 double 算准，最后转成 float 交给模型。
 *   （W10 上板时如果嫌 double 慢，再改成 float 并重新验证。）
 */

#include <math.h>
#include "features.h"

const char *FEATURE_NAMES_C[N_FEATURES] = {
    "ax_span", "ax_mean", "ax_std",
    "ay_span", "ay_mean", "ay_std",
    "az_span", "az_mean", "az_std",
    "gx_span", "gx_mean", "gx_std",
    "gy_span", "gy_mean", "gy_std",
    "gz_span", "gz_mean", "gz_std",
    "az_neg_ratio",
};

/* ---------------------------------------------------------------------
 * 10 点滑动平均（尾部对齐）
 *
 * 意思：对每个位置 i，取"它和它前面 9 个点"的平均。
 *       前面不够 10 个点（i < 9）时，有几个算几个。
 *
 * 为什么不直接对 768 个点做一次平均：
 *   那就变成一个数了，什么都分不出来。
 *   滑动平均是"沿着时间挪"，输出的还是 128 个数，
 *   但每个数都被"附近 10 个点"平滑过 —— 高频抖动被削掉，
 *   低频的摆动保留下来。之后量"跨度"才有意义。
 *
 * 为什么 n=10：100 Hz 采样，10 个点 = 0.1 秒 ≈ 10 Hz 低通。
 *   正点原子那份工程里的机械振动在 16~22 Hz，正好被压掉。
 * ------------------------------------------------------------------- */
static void moving_avg(const double *x, int len, int n, double *out)
{
    double sum = 0.0;

    for (int i = 0; i < len; i++) {
        sum += x[i];                    /* 加入当前点 */
        if (i >= n)
            sum -= x[i - n];            /* 挤掉窗口外最老的点 */

        int cnt = (i < n - 1) ? (i + 1) : n;   /* 开头不足 n 个，有几个算几个 */
        out[i] = sum / cnt;
    }
}

/* ---------------------------------------------------------------------
 * 一个轴（128 个原始值）算 3 个数写完，写进 f[base..base+2]
 *
 *   base+0  span  平滑后的跨度  = 平滑后最大值 - 平滑后最小值
 *           → 这个轴"摆得多厉害"（幅度）
 *
 *   base+1  mean  原始值的平均
 *           → 这个轴"偏向哪边"（重力方向、静态偏置）
 *
 *   base+2  std   原始值的标准差（除以 N）
 *           → 这个轴"抖得多厉害"（能量）
 *
 * 为什么 nth 只有 3 个：
 *   幅度、偏置、抖动，这三件事基本把一个轴的行为说完了。
 *   再加峰度/偏度之类，451 个样本撑不起来，只会过拟合。
 * ------------------------------------------------------------------- */
static void axis_stats(const double *x, float *f, int base, double *smooth)
{
    moving_avg(x, WIN_LEN, 10, smooth);

    double lo = smooth[0], hi = smooth[0];
    for (int i = 1; i < WIN_LEN; i++) {
        if (smooth[i] < lo) lo = smooth[i];
        if (smooth[i] > hi) hi = smooth[i];
    }

    double sum = 0.0;
    for (int i = 0; i < WIN_LEN; i++)
        sum += x[i];
    double mean = sum / WIN_LEN;

    double var = 0.0;
    for (int i = 0; i < WIN_LEN; i++) {
        double d = x[i] - mean;
        var += d * d;
    }
    var /= WIN_LEN;                     /* 除以 N，不是 N-1 */

    f[base + 0] = (float)(hi - lo);
    f[base + 1] = (float)mean;
    f[base + 2] = (float)sqrt(var);
}

void extract_features(const int16_t win[WIN_LEN][N_AXES], float f[N_FEATURES])
{
    double col[WIN_LEN];                /* 当前轴拉平成 double 数组 */
    double smooth[WIN_LEN];             /* 平滑后的结果 */

    for (int a = 0; a < N_AXES; a++) {
        for (int i = 0; i < WIN_LEN; i++)
            col[i] = (double)win[i][a];

        axis_stats(col, f, a * 3, smooth);
    }

    /* 第 19 维：az < 0 的时间点占比
     *
     * 为什么单独加这一个：
     *   加速度 z 轴测的是重力。平放时 az 是正的一大坨（约 +16500），
     *   翻个面就变成负的一大坨。而"翻面"这个动作的标志就是
     *   "一半时间是正的、一半时间是负的"。
     *   只看平均值会互相抵消（正负各一半 → 平均 ≈ 0），看不出翻面。
     *   所以这里数"有多少个点是负的"。
     */
    int neg = 0;
    for (int i = 0; i < WIN_LEN; i++)
        if (win[i][2] < 0)              /* 严格小于 0，等于 0 不算 */
            neg++;

    f[N_FEATURES - 1] = (float)neg / (float)WIN_LEN;
}
