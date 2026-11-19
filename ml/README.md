# ml/ —— PC 端数据处理与训练

**这里的东西不参与固件编译**，是拿串口抓下来的 CSV 在电脑上跑的分析脚本。
板子上跑的是 `examples/26_iot_cloud_onenet/applications/data_log.c`。

---

## 一、日常只用这两条

```powershell
cd F:\interesting\stm_project\chuanganqi\ml
chcp 65001 > $null; $env:PYTHONIOENCODING='utf-8'

python clean_data.py      # 第 1 步：CSV  ->  dataset.npz
python train_baseline.py  # 第 2 步：npz  ->  准确率 / 混淆矩阵 / 各类召回率
```

**只改模型或特征时，不用跑第 1 步。**

---

## 二、六个文件各干什么

| 文件 | 类型 | 回答什么问题 | 什么时候跑 |
|---|---|---|---|
| `clean_data.py` | **管道** | 把 5 个 CSV 切成 451 个 128×6 的窗口，存成 `dataset.npz` | 重新采了数据 / 改了 `MANIFEST` |
| `train_baseline.py` | **管道 · 主程序** | 这套特征+这个模型，准确率和每类召回是多少 | 每次改特征/改模型后 |
| `compare_models.py` | 一次性实验 | 8 个模型谁强？分组切窗有没有泄露？ | 只在 W7-A 选型时跑过，结论已进档案 |
| `explain_features.py` | 一次性实验 | 19 维特征长什么样？哪些窗口被误判？ | 想"看见"数据时跑，不训练 |
| `eval_new_rounds.py` | 一次性实验 | 补采的 2 轮有没有用？跨环境考多少分？ | 只在补数据时跑过（100% / 36.4%） |
| `probe_rotate.py` | 一次性实验 | 哪些 rotate 窗口其实没在转？ | W8-A 体检，结论：22/95 个是采集死时间 |

**"管道" = 以后还要反复跑；"一次性实验" = 结论已经写进学习档案，留着是为了能复现。**

不改名、不删。名字虽然长，但一眼能看出是干什么的。

---

## 三、数据流

```
data/*.csv  (29858 点，git 不跟踪，.gitignore 里有 /data/)
      │
      │  clean_data.py  ──  删全零行 → 按 1024 点分轮 → 轮内滑窗 128/64
      ▼                     → 剔除"静止冒充运动"的窗口
dataset.npz  (X=(451,128,6) int16, y=(451,), 295 KB)
      │
      │  train_baseline.py  ──  19 维特征 → 随机森林 300 棵 → GroupKFold 5 折
      ▼
准确率 / 混淆矩阵 / 各类召回率
```

---

## 四、一个必须记住的陷阱

`clean_data.py` 里的 `MANIFEST` 记录每个 CSV 有**几轮**（每轮 1024 点）：

```python
MANIFEST = {
    "0_static":   6,
    "1_pickup":   6,
    "2_shake_lr": 6,
    "3_flip":     6,
    "4_rotate":   7,
}
```

代码里有这么一句：

```python
arr = arr[max(0, len(arr) - rounds * 1024):]
```

**它的意思是"只要最后 N 轮"。** 所以：

> 往 CSV 里追加了数据，**一定要同时把 MANIFEST 里的轮数加上**。
> 忘了改的话，它不会报错，只会**静默地把最前面几轮丢掉**，然后你还以为数据变多了。

---

## 五、当前基线（W7-A，2026-11-19）

| 指标 | 数值 |
|---|---|
| 窗口总数 | 451（static 89 / pickup 88 / shake 89 / flip 90 / rotate 95）|
| 整体准确率 | 95.1%（GroupKFold 5 折）|
| 最弱类召回 | shake_lr 92.1% |
| rotate 召回 | 90.5%（补采前 84.6%）|
| 主要错误 | shake ↔ rotate 互相误判，占全部错误的 50% |
