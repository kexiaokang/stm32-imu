/* 自动生成，不要手改。生成脚本：ml/export_model.py --board */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#define N_CLASSES   5
#define N_FEATURES  19

/*
 * 一棵树被"拍平"成一张节点表。
 *
 *   内部节点：left / right >= 0，prob 全 0
 *   叶子节点：left == -1
 *
 * 走法（就这一个循环，整片森林都是它）：
 *
 *   int node = 0;
 *   while (t->nodes[node].left != -1) {
 *       node = (x[t->nodes[node].feature] <= t->nodes[node].threshold)
 *              ? t->nodes[node].left
 *              : t->nodes[node].right;
 *   }
 *   // 到这里 node 是叶子，概率在 t->nodes[node].prob[0..4]
 */
typedef struct {
    int16_t feature;          /* 用第几维特征比较（0~18） */
    int16_t left;             /* 左孩子下标；-1 表示叶子 */
    int16_t right;            /* 右孩子下标；-1 表示叶子 */
    float   threshold;        /* x[feature] <= threshold → 走左边 */
    float   prob[N_CLASSES];  /* 叶子：归一化后的 5 类概率；内部节点：全 0 */
} Node;

typedef struct {
    const Node *nodes;
    int         n_nodes;
    int         root;         /* 根节点下标，目前都是 0 */
} Tree;

typedef struct {
    const Tree *trees;
    int         n_trees;
} Forest;

extern const Forest FOREST_FULL;   /* 100 棵树，用全部 451 个窗口训练 */

extern const char *CLASS_NAMES[N_CLASSES];

#endif  /* MODEL_DATA_H */
