#include <rtthread.h>
#include <rtdevice.h>

#define ICM_ADDR     0x68
#define WHO_AM_I     0x75
#define PWR_MGMT_1   0x6B
#define ACCEL_XOUT_H 0x3B

static struct rt_i2c_bus_device *i2c_bus = RT_NULL;
static rt_uint32_t g_icm_err = 0;   /* 放文件顶部；循环里读失败就 g_icm_err++ 并打印

/**
 * @brief 扫描 i2c1/i2c2/i2c3 三条总线上的从机设备并打印地址
 *
 * 逐条查找已注册的 I2C 总线，对 7 位地址区间 0x08~0x77 发起空写探测，
 * 从机返回 ACK 即判定该地址上有设备在线。
 *
 * @note 仅用于调试定位传感器所在总线与地址，遍历全部地址耗时较长
 */
static void i2c_scan(void)
{
    struct rt_i2c_bus_device *bus;
    const char *names[] = { "i2c1", "i2c2", "i2c3" };                  /* 需要探测的总线设备名 */
    for (int b = 0; b < 3; b++) {
        bus = (struct rt_i2c_bus_device *)rt_device_find(names[b]);     /* 按名字查找已注册的 I2C 总线 */
        if (bus == RT_NULL) { rt_kprintf("%s: not found\n", names[b]); continue; }  /* 总线未注册，跳过 */
        rt_kprintf("scan %s:\n", names[b]);
        for (rt_uint8_t addr = 0x08; addr < 0x78; addr++) {             /* 遍历 7 位地址合法区间 0x08~0x77 */
            struct rt_i2c_msg msg = { .addr = addr, .flags = RT_I2C_WR, .buf = RT_NULL, .len = 0 };  /* 空写探测：只看从机是否 ACK */
            if (rt_i2c_transfer(bus, &msg, 1) == 1)
                rt_kprintf("  found 0x%02X\n", addr);                   /* 传输成功说明该地址上有设备应答 */
        }
    }
}
MSH_CMD_EXPORT(i2c_scan, scan all i2c buses);

/**
 * @brief 向 ICM20608 指定寄存器写入一个字节
 *
 * 将寄存器地址与待写入数据拼成两字节缓冲，通过一次 I2C 写传输
 * 发送到器件地址 ICM_ADDR（0x68），完成单寄存器写操作。
 *
 * @param reg 目标寄存器地址
 * @param val 待写入的字节数据
 *
 * @return RT_EOK 表示写入成功，-RT_ERROR 表示 I2C 传输失败
 */
static rt_err_t icm_write_reg(rt_uint8_t reg, rt_uint8_t val)
{
    /* 组包：[寄存器地址, 待写入数据] */
    rt_uint8_t buf[2] = { reg, val };
    /* 构造 I2C 写消息：目标地址 + 写标志 + 数据缓冲 */
    struct rt_i2c_msg msg = { .addr = ICM_ADDR, .flags = RT_I2C_WR, .buf = buf, .len = 2 };
    /* 同步传输 1 条消息，成功返回 RT_EOK，否则返回错误 */
    return (rt_i2c_transfer(i2c_bus, &msg, 1) == 1) ? RT_EOK : -RT_ERROR;
}

static rt_err_t icm_read_regs(rt_uint8_t reg, rt_uint8_t *buf, rt_uint8_t len)
{
    /* 两次传输组成一次"写寄存器地址 + 读数据"的复合时序，中间不发 STOP，保证地址与读操作在同一事务内 */
    struct rt_i2c_msg msgs[2] = {
        { .addr = ICM_ADDR, .flags = RT_I2C_WR, .buf = &reg, .len = 1 },   /* 第 1 条：发送要读的寄存器地址 */
        { .addr = ICM_ADDR, .flags = RT_I2C_RD, .buf = buf,  .len = len }, /* 第 2 条：从该地址连续读出 len 字节 */
    };
    /* 两条消息都完整发出才算成功，否则返回错误 */
    return (rt_i2c_transfer(i2c_bus, msgs, 2) == 2) ? RT_EOK : -RT_ERROR;
}

// MSH_CMD_EXPORT(icm_tick_test, measure real period);

static rt_err_t icm_read_retry(rt_uint8_t reg, rt_uint8_t *buf, rt_uint8_t len)
{
    for (int i = 0; i < 3; i++) {
        if (icm_read_regs(reg, buf, len) == RT_EOK) return RT_EOK;
        rt_thread_mdelay(5);
    }
    return -RT_ERROR;
}


void icm_demo(void)
{
    i2c_bus = (struct rt_i2c_bus_device *)rt_device_find("i2c1");  /* 总线名以 i2c_scan 结果为准 */
    if (i2c_bus == RT_NULL) { rt_kprintf("i2c1 not found\n"); return; }

    rt_uint8_t id = 0;                                              /* 读 WHO_AM_I 确认芯片在线（ICM20608 应为 0x68） */
    if (icm_read_regs(WHO_AM_I, &id, 1) != RT_EOK) { rt_kprintf("read fail\n"); return; }
    rt_kprintf("WHO_AM_I = 0x%02X\n", id);

    icm_write_reg(PWR_MGMT_1, 0x80);                                /* DEVICE_RESET：软复位，清空所有寄存器为默认值 */
    rt_thread_mdelay(100);                                          /* 等待复位完成 */
    icm_write_reg(PWR_MGMT_1, 0x01);                                /* CLKSEL=1：关闭睡眠并选 PLL 作为时钟源 */
    rt_thread_mdelay(10);    

    icm_write_reg(0x1A, 0x03);  /* CONFIG: DLPF 44Hz；开了 DLPF 后陀螺基频 1kHz */
    icm_write_reg(0x19, 0x09);  /* SMPLRT_DIV: ODR = 1000/(1+9) = 100Hz */
    icm_write_reg(0x1B, 0x00);  /* GYRO_CONFIG: ±250dps */
    icm_write_reg(0x1C, 0x00);  /* ACCEL_CONFIG: ±2g → 16384 LSB/g */                                       /* 等待时钟稳定后开始采样 */

    for (int i = 0; i < 20; i++) {                                  /* 连续采样 20 次用于观察数据 */
        rt_uint8_t d[6];
        if (icm_read_retry(ACCEL_XOUT_H, d, 6) == RT_EOK) {          /* 从 0x3B 起一次读 6 字节：X/Y/Z 各 2 字节高字节在前 */
            short ax = (short)((d[0] << 8) | d[1]);                 /* 拼接并转有符号：加速度 X 原始值 */
            short ay = (short)((d[2] << 8) | d[3]);                 /* Y 轴原始值 */
            short az = (short)((d[4] << 8) | d[5]);                 /* Z 轴原始值 */
            rt_kprintf("acc: %d, %d, %d\n", ax, ay, az);
        }
        else {
            g_icm_err++;                                        /* ← 新增：失败计数 */
            rt_kprintf("read fail, total err = %d\n", g_icm_err);
        }
        rt_thread_mdelay(200);                                      /* 采样间隔 200ms */
    }
}
MSH_CMD_EXPORT(icm_demo, read icm20608 accel);


void icm_tick_test(void)
{
    rt_tick_t last = rt_tick_get();
    for (int i = 0; i < 10; i++) {
        rt_thread_mdelay(200);
        rt_tick_t now = rt_tick_get();
        rt_kprintf("period = %d tick\n", now - last);   /* 潘多拉 1 tick = 1ms */
        last = now;
    }
}
MSH_CMD_EXPORT(icm_tick_test, measure real period);


static struct rt_semaphore g_smp_sem;
static rt_timer_t g_smp_timer;

static void smp_cb(void *param)
{
    rt_sem_release(&g_smp_sem);   /* 中断里只干这一件事 */
}

void icm_sample_100hz(void)
{
    i2c_bus = (struct rt_i2c_bus_device *)rt_device_find("i2c1");
    if (i2c_bus == RT_NULL) { rt_kprintf("i2c1 not found\n"); return; }

    rt_sem_init(&g_smp_sem, "smp", 0, RT_IPC_FLAG_PRIO);
    g_smp_timer = rt_timer_create("smp", smp_cb, RT_NULL,
                                  RT_TICK_PER_SECOND / 100,   /* 10ms = 100Hz */
                                  RT_TIMER_FLAG_PERIODIC);
    if (g_smp_timer == RT_NULL) { rt_kprintf("timer create fail\n"); return; }
    rt_timer_start(g_smp_timer);

    rt_uint32_t ok = 0, err = 0;
    rt_tick_t t0 = rt_tick_get();
    while (ok + err < 500) {                /* 理论 5 秒 */
        rt_sem_take(&g_smp_sem, RT_WAITING_FOREVER);
        rt_uint8_t d[6];
        if (icm_read_retry(ACCEL_XOUT_H, d, 6) == RT_EOK) ok++;
        else err++;
    }
    rt_timer_stop(g_smp_timer);
    rt_kprintf("ok=%d err=%d elapsed=%d tick\n", ok, err, rt_tick_get() - t0);
}
MSH_CMD_EXPORT(icm_sample_100hz, 100Hz sampling via timer and sem);