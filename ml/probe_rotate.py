"""W8：rotate 窗口体检 —— 找出"标了 rotate 但根本没在转"的窗口
   两个 ??? 自己填，写完跑，跑完把最后那张表贴给我
"""

import numpy as np
from pathlib import Path

NPZ = Path(__file__).resolve().parent / "dataset.npz"
THRESH = 500          # |gz 均值| 小于它 = 没在转（单位 LSB）

d = np.load(NPZ, allow_pickle=True)
X = d["X"].astype(np.float64)     # 一定要转 float，不然 int16 会溢出
y = d["y"].astype(int)

idx = np.where(y == 4)[0]         # rotate 类的所有窗口下标
print("rotate 窗口共", len(idx), "个")
print("{:>6}{:>12}{:>12}".format("窗口", "gz均值LSB", "转角(度)"))

bad = 0
for i in idx:
    gz = X[i][:, 5]               # 第 i 个窗口的 gz 一列，128 个数

    # ① 这个窗口 gz 的平均值（单位 LSB，可以带负号）
    m = gz.mean()

    # ② 这个窗口一共转了多少度
    #    每个点 = 0.01 秒；gz/131 = 当时角速度(度/秒)；累加起来就是度
    deg = gz.sum()/131*0.01

    if abs(m) < THRESH:
        bad += 1
        print("{:>6}{:>12.0f}{:>12.1f}  <-- 没在转".format(i, m, deg))
    else:
        print("{:>6}{:>12.0f}{:>12.1f}".format(i, m, deg))

print()
print("判为没在转的窗口:", bad, "/", len(idx))