#include <rtthread.h>

int g_data = 1;                     /* .data 区 */
static int g_bss;                   /* .bss 区 */

/**
 * @brief  演示内存布局与指针运算
 *
 * 打印栈、堆、.data 区、.bss 区中各类变量的地址，直观对比各存储区的位置关系；
 * 并通过遍历 short 数组验证指针宽度与指针算术的步长。
 *
 * @param  void 无参数
 * @return void 无返回值
 */
void ptr_demo(void)
{
    int local = 2;                            /* 栈 */
    static int s_data = 3;                    /* .data 区 */
    int *heap = (int *)rt_malloc(sizeof(int));/* 堆 */

    /* 依次打印各存储区变量的地址，直观对比内存布局 */
    rt_kprintf(".data &g_data = %p\n", &g_data);   /* 全局已初始化变量，位于 .data */
    rt_kprintf(".bss  &g_bss  = %p\n", &g_bss);    /* 全局未初始化变量，位于 .bss */
    rt_kprintf(".data &s_data = %p\n", &s_data);   /* 函数内 static 变量，位于 .data */
    rt_kprintf("heap  heap    = %p\n", heap);      /* 堆：rt_malloc 返回的地址 */
    rt_kprintf("stack &local  = %p\n", &local);    /* 栈：局部变量地址，通常最高 */
    rt_free(heap);                                 /* 释放堆内存，避免内存泄漏 */

    short arr[4] = {0x1111, 0x2222, 0x3333, 0x4444}; /* 局部数组，每个元素占 2 字节 */
    short *p = arr;                                   /* p 指向数组首元素 arr[0] */
    rt_kprintf("sizeof(short)=%d sizeof(int*)=%d\n", sizeof(short), sizeof(int *)); /* 验证类型与指针宽度 */
    for (int i = 0; i < 4; i++) {
        rt_kprintf("p=%p *p=0x%04x\n", p, *p);    /* 地址每次递增 2，取值依次为数组元素 */
        p++;                                       /* 指针算术：按 short 大小（2 字节）前进 */
    }
}
MSH_CMD_EXPORT(ptr_demo, memory layout and pointer arithmetic);