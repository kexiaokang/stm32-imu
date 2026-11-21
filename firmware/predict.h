/*
 * W9-1 你要实现的两个函数。
 * 数据结构在 model_data.h 里（先读那里面 Node 的注释）。
 */
#ifndef PREDICT_H
#define PREDICT_H

#include "model_data.h"

/*
 * 走完一棵树，把它这个叶子给出的 5 类概率【累加】到 out[0..4] 上。
 *
 * 参数：
 *   t    —— 一棵树（t->nodes 是节点表，t->root 是根节点下标）
 *   x    —— 19 维特征
 *   out  —— 长度 5 的累加器，由调用者负责清零
 *
 * 注意：这里只做累加，不做除以 300，也不做 argmax。
 */
void tree_predict(const Tree *t, const float *x, float *out);

/*
 * 走完整片森林，返回预测的类别（0~4）。
 *
 * 做法提示：对每棵树调用一次 tree_predict，累加出 5 个分数，
 *           最后返回分数最大的那个下标。
 *           （不用除以 n_trees —— 同时除以一个正数不改变谁最大）
 */
int forest_predict(const Forest *f, const float *x);

#endif /* PREDICT_H */
