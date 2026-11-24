/*
 * W10-3：接上真实的 ICM20608，让板子自己认动作
 * =====================================================================
 *
 * 到 W10-2 为止，板子只会"算别人递进来的题"：
 *    board_test  给 19 维特征    -> 出类别
 *    feat_test   给 128x6 原始点 -> 算特征 -> 出类别
 *
 * 这一步补上最前面那段：数据从哪来。
 *
 *    ICM20608 ──100Hz──> 128 点滑窗 ──> 19 维特征 ──> 100 棵森林 ──> 串口
 *    (i2c1,0x68)          1.28 秒        features.c      predict.c
 *
 * 为什么一步拆成三个命令（而不是直接连续跑）：
 *    imu_rate  只验证"采样对不对"（128 点是不是 1.28 秒、丢不丢点）
 *    imu_start 才做连续识别（每个窗口打印类别 + 置信度 + 耗时）
 *    imu_stop  停
 *   万一 imu_start 的输出乱了，先用 imu_rate 就能判断是不是采样层的问题。
 *
 * 为什么阈值、特征、树都不用改：
 *   训练数据就是这个板子、这个量程（±2g / ±250dps）、这个 ODR（100 Hz）采的。
 *   W10-2 已经证明"同一份原始点，板子和 PC 算出的特征一致（53 ppb）"。
 *   所以这里只要把数据通路接上，结果就该和 PC 上一模一样。
 *
 * 内存账（为什么可以放心加）：
 *    g_win  128x6 个 int16 = 1536 B  放静态区（不要放栈）
 *    识别线程栈 4096 B（extract_features 内部要 2x128 个 double = 2 KB）
 *    Flash 只多几 KB 代码；上一个存档点还剩 62.6 KB。
 *
 * 怎么用（串口 115200，勾"加回车换行"）：
 *    imu_rate          采 128 点，报耗时/采样率/丢点（先跑这个）
 *    imu_start         连续识别，每 1.28 秒打印一行
 *    imu_stop          停
 */

#include <rtthread.h>
#include <rtdevice.h>
#include <board.h>          /* CoreDebug / DWT / SystemCoreClock */

#include "model_data.h"     /* N_FEATURES / N_CLASSES / CLASS_NAMES / FOREST_FULL */
#include "predict.h"        /* tree_predict / forest_predict */
#include "features.h"       /* extract_features / WIN_LEN / N_AXES */

/* --- ICM20608 寄存器（和数据手册、data_log.c 完全一致） --- */
#define ICM_ADDR       0x68
#define WHO_AM_I       0x75
#define PWR_MGMT_1     0x6B
#define ACCEL_XOUT_H   0x3B     /* 0x3B..0x40：ax ay az，高字节在前 */
#define GYRO_XOUT_H    0x43     /* 0x43..0x48：gx gy gz */

/* --- 线程参数 --- */
#define IMU_STACK      4096
#define IMU_PRIO       15       /* 比 shell(20) 高、比 main(10) 低：采样不能被打印拖住 */

/* --- 时序平滑参数（W10-3-c）---
 * 模型每个窗口都会给出一个答案，但相邻窗口经常在 1/2/4 之间跳：
 * 因为"轻微移动"这种动作本来就落在两类的边界上。
 * 工程上不靠改模型，靠"连续几个窗口一致才认"：
 */
#define VOTE_N     3        /* 看最近 3 个窗口 */
#define VOTE_MIN   2        /* 至少 2 个一致才输出 */
#define CONF_MIN   600      /* 置信度低于 600‰(60%) 的窗口不参与投票 */

/* ---------------------------------------------------------------------
 * 全局状态
 * ------------------------------------------------------------------- */
static struct rt_i2c_bus_device *g_bus   = RT_NULL;
static struct rt_semaphore       g_sem;             /* 定时器 -> 采样线程 */
static rt_timer_t                g_timer = RT_NULL; /* 10 ms 周期 = 100 Hz */
static rt_thread_t               g_thread = RT_NULL;
static volatile int              g_run    = 0;      /* 1 = 正在连续识别 */
static int                       g_ready  = 0;      /* 硬件/信号量/定时器是否已就绪 */

/* 128 个时间点 x 6 轴。放静态区：1.5 KB，不能放栈上（线程栈只有 4 KB） */
static rt_int16_t g_win[WIN_LEN][N_AXES];

static volatile rt_uint32_t g_drop     = 0;   /* 累计丢点（等不到节拍 或 读失败）*/
static volatile rt_uint32_t g_mismatch = 0;   /* 自检：两种算法给出的类别不一致的次数 */

/* 最近 VOTE_N 个窗口的原始判定结果（环形），给"时序平滑"用 */
static int g_hist_cls[VOTE_N];
static int g_hist_conf[VOTE_N];
static int g_hist_cnt = 0;      /* 已攒了几个（最多 VOTE_N，刚启动时不足）*/
static int g_hist_pos = 0;      /* 下一个写哪里 */

/* ---------------------------------------------------------------------
 * 一、底层 I2C：读一个寄存器 / 连续读 N 个寄存器
 *
 * 和 icm20608_demo.c 的写法一样，为什么必须这样写：
 *   I2C 读寄存器 = 先"写"寄存器地址（告诉芯片我要读哪），
 *                  再"读"数据。两次传输之间【不能有 STOP】，
 *   否则芯片会把它们当成两件不相干的事。
 *
 *   所以用 rt_i2c_msg msgs[2]，一次 rt_i2c_transfer(..., 2) 发出去，
 *   中间由驱动自动插入"重复起始条件"。
 * ------------------------------------------------------------------- */
static rt_err_t imu_wr(rt_uint8_t reg, rt_uint8_t val)
{
    rt_uint8_t buf[2] = { reg, val };
    struct rt_i2c_msg msg = { .addr = ICM_ADDR, .flags = RT_I2C_WR, .buf = buf, .len = 2 };
    return (rt_i2c_transfer(g_bus, &msg, 1) == 1) ? RT_EOK : -RT_ERROR;
}

static rt_err_t imu_rd(rt_uint8_t reg, rt_uint8_t *buf, rt_uint8_t len)
{
    struct rt_i2c_msg msgs[2] = {
        { .addr = ICM_ADDR, .flags = RT_I2C_WR, .buf = &reg, .len = 1 },
        { .addr = ICM_ADDR, .flags = RT_I2C_RD, .buf = buf,  .len = len },
    };
    return (rt_i2c_transfer(g_bus, msgs, 2) == 2) ? RT_EOK : -RT_ERROR;
}

/* 失败重试 3 次：I2C 偶尔被打断是正常的，一次失败就丢点太浪费 */
static rt_err_t imu_rd_retry(rt_uint8_t reg, rt_uint8_t *buf, rt_uint8_t len)
{
    for (int i = 0; i < 3; i++) {
        if (imu_rd(reg, buf, len) == RT_EOK) return RT_EOK;
        rt_thread_mdelay(2);
    }
    return -RT_ERROR;
}

/* ---------------------------------------------------------------------
 * 二、芯片初始化 + 内核对象准备（幂等：重复调用只会做一次）
 *
 * 初始化序列和 data_log.c 一模一样，因为训练数据就是这个配置采的：
 *   任意一项改了（量程、ODR、滤波），特征分布就变了，模型会失效。
 *
 * 为什么加速度和陀螺分两次读（0x3B 和 0x43）：
 *   它们中间夹着温度寄存器 0x41~0x42。
 *   一次读 14 字节也能读，但要把温度两字节跳过去，容易写错索引；
 *   分两次读只有 6+6 字节，索引干净，代价只是多一次 I2C 事务。
 * ------------------------------------------------------------------- */
static void imu_timer_cb(void *param)
{
    rt_sem_release(&g_sem);     /* 中断上下文只做这一件事，别在这里读传感器 */
}

static int imu_prepare(void)
{
    if (g_ready) return 0;

    g_bus = (struct rt_i2c_bus_device *)rt_device_find("i2c1");
    if (g_bus == RT_NULL) {
        rt_kprintf("[imu] i2c1 not found\r\n");
        return -1;
    }

    rt_uint8_t id = 0;
    if (imu_rd_retry(WHO_AM_I, &id, 1) != RT_EOK) {
        rt_kprintf("[imu] read WHO_AM_I fail（传感器没接好/上电）\r\n");
        return -1;
    }
    rt_kprintf("[imu] WHO_AM_I = 0x%02X%s\r\n", id, (id == 0x68) ? "" : "（不是 0x68，仍继续）");

    imu_wr(PWR_MGMT_1, 0x80);       /* DEVICE_RESET：软复位，寄存器回默认值 */
    rt_thread_mdelay(100);          /* 等复位完成 */
    imu_wr(PWR_MGMT_1, 0x01);       /* 唤醒 + CLKSEL=1（用 PLL 当时钟源，更稳） */
    rt_thread_mdelay(10);
    imu_wr(0x1A, 0x03);             /* CONFIG：DLPF 44 Hz（滤掉高频抖动，训练时就是它）*/
    imu_wr(0x19, 0x09);             /* SMPLRT_DIV：1000/(1+9) = 100 Hz */
    imu_wr(0x1B, 0x00);             /* GYRO_CONFIG：±250 dps */
    imu_wr(0x1C, 0x00);             /* ACCEL_CONFIG：±2 g */

    rt_sem_init(&g_sem, "imu_s", 0, RT_IPC_FLAG_PRIO);

    g_timer = rt_timer_create("imu_t", imu_timer_cb, RT_NULL,
                              RT_TICK_PER_SECOND / 100,     /* 10 ms = 100 Hz */
                              RT_TIMER_FLAG_PERIODIC);      /* 和 data_log.c 一样用硬件定时器 */
    if (g_timer == RT_NULL) {
        rt_kprintf("[imu] timer create fail\r\n");
        return -1;
    }

    /* DWT 是 Cortex-M4 里的一个硬件周期计数器，专门用来量"这段代码跑了多少个时钟周期"。
     * 没有它，只能拿 rt_tick_get() 量，分辨率 1 ms——而一次推理只有几百 ?s，量不出来。 */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

    g_ready = 1;
    return 0;
}

/* ---------------------------------------------------------------------
 * 三、一次采样：6 个 int16
 *
 * 芯片给的是"高字节在前"（大端），板子是"低字节在前"（小端），
 * 所以必须手动拼：(d[0] << 8) | d[1]。
 * 直接 (int16_t *)buf 强转在 ARM 上会得到完全错误的数字。
 * ------------------------------------------------------------------- */
static rt_err_t imu_sample_once(rt_int16_t out[N_AXES])
{
    rt_uint8_t d[6];

    if (imu_rd_retry(ACCEL_XOUT_H, d, 6) != RT_EOK) return -RT_ERROR;
    out[0] = (rt_int16_t)((d[0] << 8) | d[1]);
    out[1] = (rt_int16_t)((d[2] << 8) | d[3]);
    out[2] = (rt_int16_t)((d[4] << 8) | d[5]);

    if (imu_rd_retry(GYRO_XOUT_H, d, 6) != RT_EOK) return -RT_ERROR;
    out[3] = (rt_int16_t)((d[0] << 8) | d[1]);
    out[4] = (rt_int16_t)((d[2] << 8) | d[3]);
    out[5] = (rt_int16_t)((d[4] << 8) | d[5]);

    return RT_EOK;
}

/* 把信号量里积压的节拍清空（上次停止时可能剩了 1 个） */
static void imu_drain_sem(void)
{
    while (rt_sem_take(&g_sem, 0) == RT_EOK) {
        /* 空循环：取一个扔一个 */
    }
}

/* 周期数 -> 微秒：主频 80 MHz 时 1 ?s 正好 80 个周期。
 * 用 SystemCoreClock 而不是写死 80000000，将来改主频不会算错。 */
static rt_uint32_t cyc_to_us(rt_uint32_t cyc)
{
    return (rt_uint32_t)((rt_uint64_t)cyc * 1000000ULL / SystemCoreClock);
}

/* ---------------------------------------------------------------------
 * 四、一个窗口 -> 一次识别 -> 一行打印
 *
 * 这里没有直接调用 forest_predict()，为什么：
 *   forest_predict() 只返回"哪一类"，不给"有多确定"。
 *   而"置信度"在演示时最有用——模型说"我 92% 确定是 shake"，
 *   比只说"shake"信息量大得多，也方便你判断动作做标准了没有。
 *
 *   所以这里自己把 100 棵树的概率累加起来（和 forest_predict 同一个算法，
 *   只是多留了 5 个桶），最后顺手再用 forest_predict 算一遍做对照：
 *   两者应该 100% 一致，不一致就说明代码被改坏了（chk=BAD）。
 *
 * 为什么置信度用千分比打印：
 *   rt_kprintf 不支持 %f。把 0.923 变成 923，再拆成 "92.3%" 打印。
 * ------------------------------------------------------------------- */
static void imu_report(int win_no, int drop)
{
    float feat[N_FEATURES];
    float prob[N_CLASSES];
    rt_uint32_t t0, cyc_feat, cyc_pred;
    int i, c, best = 0, ref, pm;
    float total = 0.0f;

    /* ① 768 个原始点 -> 19 维特征（内部是 double 累加，最后转 float） */
    t0 = DWT->CYCCNT;
    extract_features(g_win, feat);
    cyc_feat = DWT->CYCCNT - t0;

    /* ② 19 维特征 -> 100 棵树 -> 5 个概率桶 */
    for (c = 0; c < N_CLASSES; c++) prob[c] = 0.0f;
    t0 = DWT->CYCCNT;
    for (i = 0; i < FOREST_FULL.n_trees; i++)
        tree_predict(&FOREST_FULL.trees[i], feat, prob);
    cyc_pred = DWT->CYCCNT - t0;

    /* ③ 哪个桶最大就是哪一类，桶里的占比就是置信度 */
    for (c = 0; c < N_CLASSES; c++) total += prob[c];
    for (c = 1; c < N_CLASSES; c++)
        if (prob[c] > prob[best]) best = c;

    pm = (total > 0.0f) ? (int)(prob[best] / total * 1000.0f + 0.5f) : 0;

    /* ④ 自检：用正式接口再算一遍，结果必须一样 */
    ref = forest_predict(&FOREST_FULL, feat);
    if (ref != best) g_mismatch++;

    rt_kprintf("#%-3d raw: class=%d(%s) conf=%d.%d%% feat=%dus pred=%dus drop=%d chk=%s\r\n",
               win_no, best, CLASS_NAMES[best], pm / 10, pm % 10,
               (int)cyc_to_us(cyc_feat), (int)cyc_to_us(cyc_pred),
               drop, (ref == best) ? "OK" : "BAD");

    /* 诊断行：把"决定 2 还是 4"的几个关键特征打出来（W10-3-d）
     *
     * 为什么要看这几个：训练数据里 shake 和 rotate 的区别就藏在
     *   gy_span（绕 y 转多少）和 gz_span（绕 z 转多少）上：
     *     shake : ax_span≈2423  gy_span≈174   gz_span≈2004
     *     rotate: ax_span≈1781  gy_span≈1040  gz_span≈8229
     * 你边做动作边看这 5 个数，就知道自己的动作"偏到哪一类去了"，
     * 比盯着 class 猜原因快得多。
     */
    rt_kprintf("     diag: ax_span=%d az_span=%d gy_span=%d gz_span=%d az_mean=%d\r\n",
               (int)feat[0], (int)feat[6], (int)feat[12], (int)feat[15], (int)feat[7]);

    /* ⑤ 时序平滑：把"每个窗口的答案"变成"现在的动作"
     *
     * 为什么需要：
     *   窗口是硬切的 1.28 秒。手在两类动作之间轻微移动时，
     *   相邻窗口会 1 -> 2 -> 4 乱跳。人眼知道"还是同一个动作"，
     *   但单窗口输出不会——它没有记忆。
     *
     * 做法（工业部署里最常用的两种手段都用上了）：
     *   ① 置信度门槛：< 60% 的窗口直接弃权，不参与投票（它本来就在猜）
     *   ② 多数表决：最近 3 个有效窗口里，至少 2 个说同一类，才输出
     *
     * 代价：从"动作开始"到"稳定输出"最多晚 1 个窗口（1.28 秒）。
     *   想更稳就把 VOTE_N 加到 5，代价是慢 2 个窗口。
     */
    g_hist_cls[g_hist_pos]  = best;
    g_hist_conf[g_hist_pos] = pm;
    g_hist_pos = (g_hist_pos + 1) % VOTE_N;
    if (g_hist_cnt < VOTE_N) g_hist_cnt++;

    int votes[N_CLASSES] = {0};
    int csum[N_CLASSES]  = {0};

    for (int k = 0; k < g_hist_cnt; k++) {
        int kc = g_hist_cls[k];
        if (g_hist_conf[k] >= CONF_MIN) {
            votes[kc]++;
            csum[kc] += g_hist_conf[k];
        }
    }

    int stable = -1;
    for (c = 0; c < N_CLASSES; c++)
        if (votes[c] >= VOTE_MIN && (stable < 0 || votes[c] > votes[stable]))
            stable = c;

    if (stable >= 0) {
        int avg = csum[stable] / votes[stable];
        rt_kprintf("     -> 稳定输出: %s  (%d/%d 窗口一致, 平均 %d.%d%%)\r\n",
                   CLASS_NAMES[stable], votes[stable], g_hist_cnt, avg / 10, avg % 10);
    } else {
        rt_kprintf("     -> 未定（最近 %d 个窗口里没有 %d 个一致且 >=%d%%，动作做干脆些）\r\n",
                   g_hist_cnt, VOTE_MIN, CONF_MIN / 10);
    }
}

/* ---------------------------------------------------------------------
 * 五、连续识别线程
 *
 * 循环很朴素：攒满 128 点 -> 报一次 -> 再攒 128 点。
 * 因为定时器是自由跑的，所以窗口之间是连续的（不重不漏）。
 * ------------------------------------------------------------------- */
static void imu_thread_entry(void *param)
{
    int win = 0;
    (void)param;

    while (g_run) {
        int drop = 0;

        for (int i = 0; i < WIN_LEN; i++) {
            if (!g_run) break;

            /* 等定时器的 10 ms 节拍；200 ms 还没来就认为这一拍丢了 */
            if (rt_sem_take(&g_sem, rt_tick_from_millisecond(200)) != RT_EOK) {
                drop++;
                continue;
            }
            if (imu_sample_once(g_win[i]) != RT_EOK)
                drop++;
        }
        if (!g_run) break;

        g_drop += drop;
        imu_report(++win, drop);
    }

    rt_timer_stop(g_timer);
    g_thread = RT_NULL;
    rt_kprintf("[imu] stopped. windows=%d drop=%d mismatch=%d\r\n",
               win, (int)g_drop, (int)g_mismatch);
}

/* ---------------------------------------------------------------------
 * 六、三个 MSH 命令
 * ------------------------------------------------------------------- */

/* imu_rate：只验证采样层。128 点应该正好 1.28 秒 = 1280 ms。 */
static int imu_rate(void)
{
    if (imu_prepare() != 0) return 0;
    if (g_run) {
        rt_kprintf("[imu] 正在连续识别，先 imu_stop\r\n");
        return 0;
    }

    imu_drain_sem();
    rt_timer_start(g_timer);

    rt_tick_t t0 = rt_tick_get();
    int drop = 0;

    for (int i = 0; i < WIN_LEN; i++) {
        if (rt_sem_take(&g_sem, rt_tick_from_millisecond(200)) != RT_EOK) {
            drop++;
            continue;
        }
        if (imu_sample_once(g_win[i]) != RT_EOK)
            drop++;
    }

    rt_tick_t el = rt_tick_get() - t0;
    rt_timer_stop(g_timer);

    /* 采样率 = 点数 / 秒数，乘 10 保留一位小数 */
    int hz10 = (el > 0) ? (int)((rt_uint32_t)RT_TICK_PER_SECOND * WIN_LEN * 10 / el) : 0;

    rt_kprintf("[imu] %d 点耗时 %d ms（理论 %d ms），采样率约 %d.%d Hz，丢点 %d\r\n",
               WIN_LEN, (int)el, (int)(WIN_LEN * 1000 / 100), hz10 / 10, hz10 % 10, drop);

    /* 顺带看看数据是不是活的（全 0 或全一样 = I2C 读到的不是真数据） */
    rt_int16_t amin[3], amax[3], gmin[3], gmax[3];
    for (int a = 0; a < 3; a++) {
        amin[a] = amax[a] = g_win[0][a];
        gmin[a] = gmax[a] = g_win[0][a + 3];
    }
    for (int i = 1; i < WIN_LEN; i++) {
        for (int a = 0; a < 3; a++) {
            if (g_win[i][a]     < amin[a]) amin[a] = g_win[i][a];
            if (g_win[i][a]     > amax[a]) amax[a] = g_win[i][a];
            if (g_win[i][a + 3] < gmin[a]) gmin[a] = g_win[i][a + 3];
            if (g_win[i][a + 3] > gmax[a]) gmax[a] = g_win[i][a + 3];
        }
    }
    rt_kprintf("  acc  x[%d..%d] y[%d..%d] z[%d..%d]\r\n",
               amin[0], amax[0], amin[1], amax[1], amin[2], amax[2]);
    rt_kprintf("  gyro x[%d..%d] y[%d..%d] z[%d..%d]\r\n",
               gmin[0], gmax[0], gmin[1], gmax[1], gmin[2], gmax[2]);
    rt_kprintf("  （静止时 acc z 应该在 16384 附近；晃动时 gyro 会到几千上万）\r\n");

    return 0;
}
MSH_CMD_EXPORT(imu_rate, W10-3 sample 128 points and check 100Hz);

/* imu_start：连续识别 */
static int imu_start(void)
{
    if (imu_prepare() != 0) return 0;

    if (g_run) {
        rt_kprintf("[imu] already running\r\n");
        return 0;
    }
    if (g_thread != RT_NULL) {
        rt_kprintf("[imu] 上一个线程还没退干净，等 1 秒再 imu_start\r\n");
        return 0;
    }

    imu_drain_sem();
    g_hist_cnt = 0;         /* 清空平滑历史：每次启动都从"没有记忆"开始 */
    g_hist_pos = 0;
    g_run = 1;

    g_thread = rt_thread_create("imu", imu_thread_entry, RT_NULL,
                                IMU_STACK, IMU_PRIO, 20);
    if (g_thread == RT_NULL) {
        g_run = 0;
        rt_kprintf("[imu] thread create fail\r\n");
        return 0;
    }

    rt_thread_startup(g_thread);
    rt_timer_start(g_timer);

    rt_kprintf("[imu] 开始连续识别：每 1.28 秒一个窗口\r\n");
    rt_kprintf("[imu] 依次做：静止 3 秒 -> 拿起 -> 左右晃 -> 翻面 -> 旋转\r\n");
    return 0;
}
MSH_CMD_EXPORT(imu_start, W10-3 start realtime recognition);

/* imu_stop：停止（线程自己退出，不在这里硬删，避免删到一半） */
static int imu_stop(void)
{
    if (!g_run) {
        rt_kprintf("[imu] not running\r\n");
        return 0;
    }
    g_run = 0;
    rt_sem_release(&g_sem);     /* 把可能正在等待的线程叫醒，让它看到 g_run == 0 */
    rt_kprintf("[imu] stopping...\r\n");
    return 0;
}
MSH_CMD_EXPORT(imu_stop, W10-3 stop realtime recognition);
