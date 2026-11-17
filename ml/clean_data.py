"""
W6-A：数据清洗 + 滑窗切分
=====================================================================
输入：../data/{0_static,1_pickup,2_shake_lr,3_flip,4_rotate}.csv
输出：dataset.npz   X=(N,128,6) int16 原始值   y=(N,) int8 标签   classes=类别名

清洗做了 4 件事：
  1. 删全零行        —— I2C 读回全 0 的坏数据
  2. 按 1024 点分轮   —— 每轮是一次独立采集，轮与轮之间隔了约 3 秒
  3. 轮内滑窗 128/64  —— 窗口不跨轮，避免把 3 秒的断档切进样本
  4. 剔除"静止冒充运动"的窗口 —— 每轮开头的磨蹭段

注意：第 4 步的判据用【平滑后】的跨度，但存进 X 的是【原始值】。
      滤波只用来"判断"，不作为模型输入——否则高频信息就被丢掉了。
"""

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE / "dataset.npz"

# 每个文件的完整轮数，来自 W5-A 的采集记录
MANIFEST = {
    "0_static":   6,   # 6144 点
    "1_pickup":   6,   # 6144 点
    "2_shake_lr": 6,   # 6142 点，最后一轮少 2 个点
    "3_flip":     6,   # 6307 点，开头有一段截断的残段
    "4_rotate":   5,   # 5120 点（删过 1 轮坏数据）
}

CLASSES = list(MANIFEST.keys())
WIN = 128          # 一个窗口 128 点 = 1.28 秒
STRIDE = 64        # 步长 64 点 = 50% 重叠
SMOOTH = 10        # 10 点滑动平均 ≈ 10Hz 低通
STATIC_T = 800.0   # 四轴平滑跨度都小于它 => 这一窗其实是"静止"


def load_csv(name):
    """读 csv，返回 (N,6) int32 + 被丢弃的坏行数。

    不能直接用 np.loadtxt：串口接收偶尔会把一行截断（列数不对），
    loadtxt 遇到这种行会直接抛异常。所以逐行解析，只收严格 6 列的。
    """
    path = DATA / f"{name}.csv"
    rows, bad = [], 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 6:
                bad += 1              # 被截断的行 只收严格 6 列的
                continue
            try:
                rows.append([int(p) for p in parts])
            except ValueError:
                continue              # 表头 "ax,ay,az,gx,gy,gz"
    return np.array(rows, dtype=np.int32), bad


def drop_zero_rows(arr):
    """删掉 6 个值全为 0 的行（I2C 读失败的坏数据）。"""
    bad = np.all(arr == 0, axis=1)
    return arr[~bad], int(bad.sum())


def moving_avg(x, n=SMOOTH):
    """10 点滑动平均（尾部对齐，开头用"已有数据的平均"）。

    注意不能用 np.convolve(mode='full')——那样开头会被 0 填充，
    第一个点变成 x[0]/10，跨度直接虚高到 16000。这里用累加和实现，
    开头几个点取已有点的均值，和 check.ps1 的算法完全一致。
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.cumsum(np.insert(x, 0, 0.0))         # 前缀和，前面补个 0
    out = (c[n:] - c[:-n]) / n              # 从第 n 个点起：完整窗口   错位相减
    head = c[1:n] / np.arange(1, n)         # 前 n-1 个点：部分窗口
    return np.concatenate([head, out])


def window_span(win):
    """一个窗口里四个轴各自的平滑跨度，返回 (ax, ay, az, gz)。"""
    spans = []
    for col in (0, 1, 2, 5):              # ax ay az gz
        s = moving_avg(win[:, col].astype(np.float64))
        spans.append(float(s.max() - s.min()))
    return spans


def main():
    X, y, report = [], [], [] # ① 三个空篮子

    for label, name in enumerate(CLASSES):# ② 5个类，一个个处理
        rounds = MANIFEST[name]
        raw, n_bad = load_csv(name)
        arr, n_zero = drop_zero_rows(raw)

        # 如果点数比"应有轮数"还多，多出来的就是开头那段被截断的残段，跳掉
        offset = max(0, len(arr) - rounds * 1024) #3filp 把开头多余的点跳掉
        arr = arr[offset:]

        total_win = kept_win = dropped = 0  # 生成窗数、保留数、剔除数
        drop_span = []                         # 记录被剔除窗口的跨度，供报告用

        for r in range(rounds):
            seg = arr[r * 1024: (r + 1) * 1024]
            if len(seg) < WIN:            # 最后一轮太短就跳过
                continue
            starts = range(0, len(seg) - WIN + 1, STRIDE)# 所有窗口起点 跳64
            for s0 in starts:
                win = seg[s0: s0 + WIN]
                total_win += 1

                # 0_static 本来就该静止，不做剔除
                if label != 0:  # static 类不剔除
                    span = window_span(win) # 算这个窗口四轴的平滑跨度
                    if max(span) < STATIC_T: # 四轴跨度都太小 → 其实是静止
                        dropped += 1
                        drop_span.append(max(span))
                        continue # 丢弃这个窗口

                kept_win += 1
                X.append(win.astype(np.int16))
                y.append(label)

        report.append(
            dict(name=name, label=label, rows=len(arr), bad=n_bad, zero=n_zero,
                 total=total_win, kept=kept_win, dropped=dropped,
                 droprate=(dropped / total_win * 100) if total_win else 0.0,
                 max_drop_span=max(drop_span) if drop_span else 0.0)
        )

    X = np.stack(X)  # ③ 攒好的窗口堆成数组
    y = np.array(y, dtype=np.int8)
    np.savez_compressed(OUT, X=X, y=y, classes=np.array(CLASSES))

    # ---------------- 报告 ----------------
    print(f"{'类':<12}{'有效点':>8}{'坏行':>6}{'全零':>6}{'生成窗':>8}{'保留':>7}{'剔除':>7}{'剔除率':>8}{'剔除窗最大跨度':>16}")
    print("-" * 80)
    for r in report:
        print(f"{r['name']:<12}{r['rows']:>8}{r['bad']:>6}{r['zero']:>6}{r['total']:>8}"
              f"{r['kept']:>7}{r['dropped']:>7}{r['droprate']:>7.1f}%{r['max_drop_span']:>16.0f}")
    print("-" * 80)
    print(f"{'合计':<12}{'':>8}{sum(r['bad'] for r in report):>6}"
          f"{sum(r['zero'] for r in report):>6}{sum(r['total'] for r in report):>8}"
          f"{len(y):>7}{sum(r['dropped'] for r in report):>7}")

    print(f"\nX shape = {X.shape}   dtype = {X.dtype}")
    print(f"y 分布  = {np.bincount(y, minlength=len(CLASSES)).tolist()}")
    print(f"已保存  -> {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
