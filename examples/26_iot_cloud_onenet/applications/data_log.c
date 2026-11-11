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
static rt_err_t imu_sample_once(struct imu_sample *s)
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
    rt_sem_release(&g_tick_sem);
}

/* 采样线程：等节拍 -> 读传感器 -> 写环形缓冲 */
static void log_thread(void *param)
{
    struct imu_sample s;

    while (s_tick < s_target && !s_abort)
    {
        if (rt_sem_take(&g_tick_sem, rt_tick_from_millisecond(200)) != RT_EOK) continue;

        if (s_wr - s_rd >= RB_CAP) { s_ovf++; continue; }   /* 缓冲满：丢点 */

        if (imu_sample_once(&s) == RT_EOK) s_rb[s_wr++ % RB_CAP] = s;
        else s_ovf++;
    }

    rt_timer_stop(g_log_timer);
    s_finish = 1;
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

/* msh 命令：imulog <点数>
 * 例：imulog 1024  -> 采 10.24 秒后自动导出 CSV
 */
void imulog(int argc, char **argv)
{
    rt_uint32_t n = RB_CAP;

    if (argc > 1) n = (rt_uint32_t)atoi(argv[1]);
    if (n == 0 || n > RB_CAP) { rt_kprintf("usage: imulog <1..%d>\n", RB_CAP); return; }

    if (s_bus == RT_NULL)
    {
        s_bus = (struct rt_i2c_bus_device *)rt_device_find("i2c1");
        if (s_bus == RT_NULL) { rt_kprintf("i2c1 not found\n"); return; }
        if (imu_init() != RT_EOK) { rt_kprintf("imu init fail\n"); s_bus = RT_NULL; return; }
        rt_kprintf("imu ready, odr=100Hz\n");
    }

    if (!s_inited)
    {
        rt_sem_init(&g_tick_sem, "tick", 0, RT_IPC_FLAG_PRIO);
        g_log_timer = rt_timer_create("log", log_cb, RT_NULL,
                                      RT_TICK_PER_SECOND / 100, RT_TIMER_FLAG_PERIODIC);
        if (g_log_timer == RT_NULL) { rt_kprintf("timer fail\n"); return; }
        s_inited = RT_TRUE;
    }

    s_wr = 0; s_rd = 0; s_tick = 0; s_ovf = 0;
    s_finish = 0; s_abort = 0; s_target = n;

    rt_timer_start(g_log_timer);

    rt_thread_t tid = rt_thread_create("logs", log_thread, RT_NULL, 2048, 10, 10);
    if (tid == RT_NULL) { rt_timer_stop(g_log_timer); rt_kprintf("thread fail\n"); return; }
    rt_thread_startup(tid);

    /* 等采完；超时保护：n 点最多等 n*20ms */
    rt_int32_t guard = (rt_int32_t)(n * 20);
    while (!s_finish && guard-- > 0) rt_thread_mdelay(1);

    if (!s_finish)
    {
        s_abort = 1;
        rt_timer_stop(g_log_timer);
        rt_kprintf("timeout\n");
        rt_thread_mdelay(20);
    }

    dump_csv();
    rt_kprintf("CSV END, points=%d ovf=%d\n", (int)s_wr, (int)s_ovf);
}
MSH_CMD_EXPORT(imulog, sample IMU at 100Hz and dump CSV);
