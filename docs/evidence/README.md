# docs/evidence —— 实验证据截图

这个目录存放**每个实验做完时的屏幕实拍/串口截图**，用来证明结论是真的跑出来的，
不是事后编的。面试或答辩时可以直接翻这几张图。

**为什么单独放在 `evidence/` 而不是 `docs/` 根目录：**
`docs/` 下的 PDF 手册、原理图、figures 一共 130+ MB，不进仓库；
只有这里（总共约 120 KB）进仓库，因为它是本项目的**成果证据**。

---

## 现有证据

| 文件 | 实验 | 证明什么 |
|---|---|---|
| `01_icm_tick_test.png` | `icm_tick_test` 命令 | `rt_thread_mdelay(200)` 的实际周期是 201 tick —— 延时从上一轮**结束**算起，误差会累积 |
| `02_ptr_demo.png` | `ptr_demo` 命令 | 真板子上的内存布局：RT-Thread 线程栈是从堆里分配的，不在 RAM 顶部 |
| `03_icm_sample_100hz.png` | `icm_sample_100hz` 命令 | 定时器 + 信号量的 100Hz 采样：500 次采样理论 5 秒、实测 5001 毫秒，偏差 1ms（0.02%） |

对应代码：`examples/26_iot_cloud_onenet/applications/icm20608_demo.c`
对应记录：`F:\obsidian\研究生自我规划\学习档案.md` 的「2026-11-08 · W4 收账」一节

---

## 以后新增图片的命名规则

```
序号_实验名.png        例：04_imulog_sample.png
```

序号接着往下排，不要覆盖旧的。加图时顺手在上面的表格里补一行。
