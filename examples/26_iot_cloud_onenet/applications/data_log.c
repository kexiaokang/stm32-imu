#include <rtthread.h>
#include <rtdevice.h>
#include <stdlib.h>

#define ICM_ADDR      0x68
#define WHO_AM_I      0x75
#define PWR_MGMT_1    0x6B
#define ACCEL_XOUT_H  0x3B
#define GYRO_XOUT_H   0x43

#define RB_CAP        1024      /* 1024 点 = 10.24 秒 @100Hz，占 12KB RAM */

struct imu_sample
{
    rt_int16_t ax, ay, az;
    rt_int16_t gx, gy, gz;
};

static struct imu_sample s_rb[RB_CAP];
static volatile rt_uint32_t s_wr;        /* 写序号（累计，不回绕） */
static volatile rt_uint32_t s_rd;        /* 读序号（dump 时推进） */
static volatile rt_uint32_t s_tick;      /* 定时器已发节拍数 */
static volatile rt_uint32_t s_target;    /* 本次目标点数 */
static volatile rt_uint32_t s_ovf;       /* 丢点计数：读失败或缓冲满 */
static volatile rt_uint32_t s_finish;    /* 采样完成标志 */
static volatile rt_uint32_t s_abort;     /* 超时中止标志 */
static rt_bool_t s_inited = RT_FALSE;
static struct rt_semaphore g_tick_sem;   /* 定时器 -> 采样线程 */
static rt_timer_t g_log_timer;
static struct rt_i2c_bus_device *s_bus = RT_NULL;

static rt_err_t imu_wr(rt_uint8_t reg, rt_uint8_t val)
{
    rt_uint8_t buf[2] = { reg, val };
    struct rt_i2c_msg msg = { .addr = ICM_ADDR, .flags = RT_I2C_WR, .buf = buf, .len = 2 };
    return (rt_i2c_transfer(s_bus, &msg, 1) == 1) ? RT_EOK : -RT_ERROR;
}

static rt_err_t imu_rd(rt_uint8_t reg, rt_uint8_t *buf, rt_uint8_t len)
{
    struct rt_i2c_msg msgs[2] =
    {
        { .addr = ICM_ADDR, .flags = RT_I2C_WR, .buf = &reg, .len = 1 },
        { .addr = ICM_ADDR, .flags = RT_I2C_RD, .buf = buf,  .len = len },
    };
    return (rt_i2c_transfer(s_bus, msgs, 2) == 2) ? RT_EOK : -RT_ERROR;
}

static rt_err_t imu_rd_retry(rt_uint8_t reg, rt_uint8_t *buf, rt_uint8_t len)
{
    for (int i = 0; i < 3; i++)
    {
        if (imu_rd(reg, buf, len) == RT_EOK) return RT_EOK;
        rt_thread_mdelay(2);
    }
    return -RT_ERROR;
}

/* 芯片初始化：复位 -> 唤醒 -> 100Hz + ±2g + ±250dps */
static rt_err_t imu_init(void)
{
    rt_uint8_t id = 0;
    if (imu_rd_retry(WHO_AM_I, &id, 1) != RT_EOK) return -RT_ERROR;
    imu_wr(PWR_MGMT_1, 0x80);   /* DEVICE_RESET */
    rt_thread_mdelay(100);
    imu_wr(PWR_MGMT_1, 0x01);   /* 唤醒，CLKSEL=1 */
    rt_thread_mdelay(10);
    imu_wr(0x1A, 0x03);         /* CONFIG: DLPF 44Hz */
    imu_wr(0x19, 0x09);         /* SMPLRT_DIV: ODR = 1000/(1+9) = 100Hz */
    imu_wr(0x1B, 0x00);         /* GYRO_CONFIG: ±250dps */
    imu_wr(0x1C, 0x00);         /* ACCEL_CONFIG: ±2g */
    return RT_EOK;
}

/* 一次采样：加速度 6 字节 + 陀螺 6 字节，分两次读（避开温度寄存器 0x41-0x42） */
static rt_err_t imu_sample_once(struct imu_sample *s)//单次采样
{
    rt_uint8_t d[6];

    if (imu_rd_retry(ACCEL_XOUT_H, d, 6) != RT_EOK) return -RT_ERROR;
    s->ax = (rt_int16_t)((d[0] << 8) | d[1]);
    s->ay = (rt_int16_t)((d[2] << 8) | d[3]);
    s->az = (rt_int16_t)((d[4] << 8) | d[5]);

    if (imu_rd_retry(GYRO_XOUT_H, d, 6) != RT_EOK) return -RT_ERROR;
    s->gx = (rt_int16_t)((d[0] << 8) | d[1]);
    s->gy = (rt_int16_t)((d[2] << 8) | d[3]);
    s->gz = (rt_int16_t)((d[4] << 8) | d[5]);

    return RT_EOK;
}

/* 定时器回调：中断上下文，只发信号量 */
static void log_cb(void *param)
{
    s_tick++;
    rt_sem_release(&g_tick_sem);    // 释放信号量，叫醒采样线程
}

/* 采样线程：等节拍 -> 读传感器 -> 写环形缓冲 */
static void log_thread(void *param)     //采样线程
{
    struct imu_sample s;

    while (s_tick < s_target && !s_abort)
    {       // 1. 等信号量（等定时器滴答），最多等 200ms
        if (rt_sem_take(&g_tick_sem, rt_tick_from_millisecond(200)) != RT_EOK) continue;
            // 2. 缓冲满了就丢点
        if (s_wr - s_rd >= RB_CAP) { s_ovf++; continue; }   /* 缓冲满：丢点 */
            // 3. 读传感器，成功就写入环形缓冲
        if (imu_sample_once(&s) == RT_EOK) s_rb[s_wr++ % RB_CAP] = s;
        else s_ovf++;
    }

    rt_timer_stop(g_log_timer);// 采够了，停定时器
    s_finish = 1;              // 通知主线程"我完成了"
    rt_kprintf("sampling done: %d pts, ovf=%d\n", (int)s_wr, (int)s_ovf);
}

static void dump_csv(void)
{
    rt_kprintf("ax,ay,az,gx,gy,gz\n");
    while (s_rd != s_wr)
    {
        struct imu_sample *p = &s_rb[s_rd % RB_CAP];
        rt_kprintf("%d,%d,%d,%d,%d,%d\n", p->ax, p->ay, p->az, p->gx, p->gy, p->gz);
        s_rd++;
    }
}

/**
 * @brief  msh 命令：以 100Hz 采样 IMU 指定点数，采完后自动导出 CSV
 *
 * 首次调用会自动查找 i2c1 并初始化 ICM 传感器；采样由周期性定时器发节拍、
 * 独立线程读传感器并写入环形缓冲，本函数只负责等待完成与导出数据。
 * 例：imulog 1024 -> 采 10.24 秒后自动导出 CSV
 *
 * @param  argc  参数个数
 * @param  argv  argv[1] 为目标采样点数，范围 1..RB_CAP，缺省为 RB_CAP
 *
 * @return 无。结果通过 rt_kprintf 输出；参数非法、i2c1 不存在、
 *         传感器初始化失败、线程创建失败或等待超时会提前返回
 */
void imulog(int argc, char **argv)
{
    rt_uint32_t n = RB_CAP;                 // 默认采满整个环形缓冲

    if (argc > 1) n = (rt_uint32_t)atoi(argv[1]);   // 取命令行指定的点数
    if (n == 0 || n > RB_CAP) { rt_kprintf("usage: imulog <1..%d>\n", RB_CAP); return; }  // 点数必须落在 1..RB_CAP

    if (s_bus == RT_NULL)                   // 首次调用：查找 I2C 总线并初始化传感器
    {
        s_bus = (struct rt_i2c_bus_device *)rt_device_find("i2c1");
        if (s_bus == RT_NULL) { rt_kprintf("i2c1 not found\n"); return; }
        if (imu_init() != RT_EOK) { rt_kprintf("imu init fail\n"); s_bus = RT_NULL; return; }  // 失败则清空句柄以便下次重试
        rt_kprintf("imu ready, odr=100Hz\n");
    }

    if (!s_inited)                          // 首次调用：创建 IPC 与 100Hz 周期定时器
    {
        rt_sem_init(&g_tick_sem, "tick", 0, RT_IPC_FLAG_PRIO);
        g_log_timer = rt_timer_create("log", log_cb, RT_NULL,
                                      RT_TICK_PER_SECOND / 100, RT_TIMER_FLAG_PERIODIC);  // 10ms 周期 = 100Hz
        if (g_log_timer == RT_NULL) { rt_kprintf("timer fail\n"); return; }
        s_inited = RT_TRUE;
    }

    s_wr = 0; s_rd = 0; s_tick = 0; s_ovf = 0;     // 复位读写序号与统计量
    s_finish = 0; s_abort = 0; s_target = n;       // 清标志并设定本次目标点数

    rt_timer_start(g_log_timer);            // 先起节拍源，再拉起采样线程

    rt_thread_t tid = rt_thread_create("logs", log_thread, RT_NULL, 2048, 10, 10);  // 栈 2KB，优先级 10，时间片 10
    if (tid == RT_NULL) { rt_timer_stop(g_log_timer); rt_kprintf("thread fail\n"); return; }
    rt_thread_startup(tid);

    /* 等采完；超时保护：n 点最多等 n*20ms */
    rt_int32_t guard = (rt_int32_t)(n * 20);      // 每点预留 20ms 余量
    while (!s_finish && guard-- > 0) rt_thread_mdelay(1);

    if (!s_finish)                          // 超时：通知采样线程退出并停表
    {
        s_abort = 1;
        rt_timer_stop(g_log_timer);
        rt_kprintf("timeout\n");
        rt_thread_mdelay(20);               // 留出时间让采样线程看到 abort 后退出
    }

    dump_csv();                             // 导出已采集数据
    rt_kprintf("CSV END, points=%d ovf=%d\n", (int)s_wr, (int)s_ovf);  // 打印总点数与丢点数
}
MSH_CMD_EXPORT(imulog, sample IMU at 100Hz and dump CSV);
