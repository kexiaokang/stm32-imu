/*
 * Copyright (c) 2006-2018, RT-Thread Development Team
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2018-09-01     ZYLX         first implementation
 */

#include <rtthread.h>
#include <rtdevice.h>
#include <wlan_mgnt.h>
#include "wifi_config.h"
#include <stdlib.h>
#include <onenet.h>
#include <string.h>
#include <ap3216c.h>
#include <easyflash.h>
#include <drv_gpio.h>

#include <drv_lcd.h>
#include <rttlogo.h>
#include <board.h>

#include <aht10.h>
#define DBG_TAG "main"
#define DBG_LVL DBG_LOG
#include <rtdbg.h>
#define ONENET_TOPIC_DP "$sys/" ONENET_INFO_PROID "/" ONENET_INFO_DEVID "/thing/property/post"
//定义传感器变量(全局变量)
ap3216c_device_t ap3216c_dev;
aht10_device_t aht10_dev;

#define I2C_BUS_NAME "i2c1"
#define LED_PIN PIN_LED_R
#define NUMBER_OF_UPLOADS 100

ap3216c_device_t dev;
static void onenet_cmd_rsp_cb(uint8_t *recv_data, size_t recv_size, uint8_t **resp_data, size_t *resp_size);



void lcd_display(void *parameter)
    {
    /* 清屏 */
    lcd_clear(WHITE);

    /* 显示 RT-Thread logo */
    lcd_show_image(0, 0, 240, 69, image_rttlogo);

    /* 设置背景色和前景色 */
    lcd_set_color(WHITE, BLACK);

    /* 打印结构体信息 */
    char buf[64];
    int y = 69; // 起始 Y 坐标

    snprintf(buf, sizeof(buf), "Light: %d", g_device_status.light);
    lcd_show_string(10, y, 16, buf);
    y += 16;

    snprintf(buf, sizeof(buf), "Temperature: %d", g_device_status.temperature);
    lcd_show_string(10, y, 16, buf);
    y += 16;

    snprintf(buf, sizeof(buf), "Wet: %d", g_device_status.wet);
    lcd_show_string(10, y, 16, buf);
    y += 16;

    /* 可选：画一条分割线 */
    lcd_draw_line(0, y, 240, y);


    return;
}
    MSH_CMD_EXPORT(lcd_display, Start LCD display thread);//导出lcd_display函数，用于在RT-Thread中调用



// int main(void)
// {
//     /* 初始化 aht10传感器 用于获取温湿度*/
//     const char *i2c_bus_name = "i2c2";
//     aht10_dev = aht10_init(i2c_bus_name);
//     if (aht10_dev == RT_NULL)
//     {
//         log_e("AHT10 sensor init failed");
//         return 0;
//     }
//     /* 初始化 ap3216c传感器  用于获取光照*/
//     ap3216c_dev = ap3216c_init(I2C_BUS_NAME);
//     if (ap3216c_dev == RT_NULL)
//     {
//         log_e("AP3216C sensor init failed");
//         return 0;
//     }

//     rt_pin_mode(LED_PIN, PIN_MODE_OUTPUT);

//     /* 初始化 ap3216c */
//     dev = ap3216c_init(I2C_BUS_NAME);
//     if (dev == RT_NULL)
//     {
//         LOG_E(" The sensor initializes failure");
//         return 0;
//     }


    
//     /* 初始化 wlan 自动连接 */
//     wlan_autoconnect_init();
//     /* 使能 wlan 自动连接 */
//     rt_wlan_config_autoreconnect(RT_TRUE);

//     return 0;
// }

    
int main(void)
{
    /* 1. 初始化传感器 */
    ap3216c_dev = ap3216c_init(I2C_BUS_NAME);
    if (ap3216c_dev == RT_NULL)
    {
        log_e("AP3216C sensor init failed");
        return 0;
    }

    const char *i2c_bus_name = "i2c2";
    aht10_dev = aht10_init(i2c_bus_name);
    if (aht10_dev == RT_NULL)
    {
        log_e("AHT10 sensor init failed");
        return 0;
    }
    
    rt_pin_mode(LED_PIN, PIN_MODE_OUTPUT);

    /* 2. 初始化网络 */
    wlan_autoconnect_init();
    rt_wlan_config_autoreconnect(RT_TRUE);

    /* 3. 打印任务要求的串口信息 */
    rt_kprintf("\r\n=== MY CODE OK ===\r\n");

    /* 4. 死循环：读数据 -> 显示 -> 延时 */
    while (1)
    {
        /* 先把读到的数据存进结构体 */
        g_device_status.light = (int)ap3216c_read_ambient_light(ap3216c_dev);//通过读取ap3216dev来获取亮度数值，放入结构体g_device_status.light，下面再通过lcd_display函数显示到LCD上
        g_device_status.temperature = (int)aht10_read_temperature(aht10_dev);//通过读取aht10dev来获取温度数值
        g_device_status.wet = (int)aht10_read_humidity(aht10_dev);//通过读取aht10dev来获取湿度数值

        /* 然后再调用显示函数 */
        lcd_display(RT_NULL);//调用lcd_display函数，显示结构体信息

        /* 延时 2 秒 */
        rt_thread_mdelay(5000);
    }
    rt_kprintf("temp=%d, wet=%d, light=%d\r\n",
           g_device_status.temperature,
           g_device_status.wet,
           g_device_status.light);
    return 0;
}


// static void onenet_upload_entry(void *parameter)
// {
//     int value = 0;
//     int i = 0;

//     /* 往 light 数据流上传环境光数据 */
//     for (i = 0; i < NUMBER_OF_UPLOADS; i++)
//     {
//         value = (int)ap3216c_read_ambient_light(dev);

//         if (onenet_mqtt_upload_digit("light", value) < 0)
//         {
//             LOG_E("upload has an error, stop uploading");
//             break;
//         }
//         else
//         {
//             LOG_D("buffer : {\"light\":%d}", value);
//         }

//         rt_thread_mdelay(5 * 1000);
//     }
// }




rt_err_t onenet_mqtt_upload_multi_digit(const char **ds_names, const double *digits, size_t count)
{
    if (!ds_names || !digits || count == 0)
        return -RT_ERROR;

    rt_err_t result = RT_EOK;
    cJSON *root = cJSON_CreateObject();
    cJSON *params = cJSON_CreateObject();
    char *msg_str = RT_NULL;
    size_t length = 0;
    uint8_t *send_buffer = RT_NULL;

    if (!root || !params)
    {
        log_e("Create cJSON object failed!");
        result = -RT_ENOMEM;
        goto __exit;
    }

    cJSON_AddItemToObject(root, "id", cJSON_CreateString("123"));
    cJSON_AddItemToObject(root, "params", params);

    for (size_t i = 0; i < count; i++)
    {
        if (!ds_names[i])
            continue;

        cJSON *temp = cJSON_CreateObject();
        if (!temp)
        {
            log_e("Create cJSON temp object failed for %s", ds_names[i]);
            continue;
        }
        cJSON_AddNumberToObject(temp, "value", digits[i]);
        cJSON_AddItemToObject(params, ds_names[i], temp);
    }

    msg_str = cJSON_PrintUnformatted(root);
    if (!msg_str)
    {
        log_e("cJSON print unformatted failed!");
        result = -RT_ENOMEM;
        goto __exit;
    }

    send_buffer = ONENET_MALLOC(strlen(msg_str));
    if (!send_buffer)
    {
        log_e("Malloc send buffer failed!");
        result = -RT_ENOMEM;
        goto __exit;
    }

    strncpy((char *)send_buffer, msg_str, strlen(msg_str));
    length = strlen(msg_str);

    if (onenet_mqtt_publish(ONENET_TOPIC_DP, send_buffer, length) < 0)
    {
        log_e("MQTT publish failed!");
        result = -RT_ERROR;
    }

__exit:
    if (send_buffer)
        ONENET_FREE(send_buffer);
    if (msg_str)
        cJSON_free(msg_str);
    if (root)
        cJSON_Delete(root);

    return result;
}


static void onenet_upload_entry(void *parameter)
{
    int i = 0;
    int light = 0, temperature = 0, wet = 0;
    
    const char *ds_names[] = {"light", "temperature", "wet"};
    double ds_values[3];

    for (i = 0; i < NUMBER_OF_UPLOADS; i++)
    {
        // 读取传感器数据
        light = (int)ap3216c_read_ambient_light(ap3216c_dev);  //通过读取ap3216dev来获取亮度数值
        temperature = (int)aht10_read_temperature(aht10_dev);  //通过读取aht10dev来获取温度数值
        wet = (int)aht10_read_humidity(aht10_dev);             //通过读取aht10dev来获取湿度数值

        ds_values[0] = light;
        ds_values[1] = temperature;
        ds_values[2] = wet;

        // 一次性上传三个数据流
        if (onenet_mqtt_upload_multi_digit(ds_names, ds_values, 3) < 0)
        {
            log_e("Upload has an error, stop uploading");
            break;
        }
        else
        {
            log_d("buffer : {\"light\":%d, \"temperature\":%d, \"wet\":%d}", light, temperature, wet);
        }

        // 每5秒上传一次
        rt_thread_mdelay(5 * 1000);
    }
}

void onenet_upload_cycle(void)
{
    rt_thread_t tid;

    /* 设置 onenet 回调响应函数 */
    onenet_set_cmd_rsp_cb(onenet_cmd_rsp_cb);

    /* 传创建线程 */
    tid = rt_thread_create("onenet_send",
                           onenet_upload_entry,
                           RT_NULL,
                           2 * 1024,
                           RT_THREAD_PRIORITY_MAX / 3 - 1,
                           5);
    if (tid)
    {
        rt_thread_startup(tid);
    }
}
MSH_CMD_EXPORT(onenet_upload_cycle, upload data to onenet);

rt_err_t onenet_port_save_device_info(char *dev_id, char *api_key)
{
    EfErrCode err = EF_NO_ERR;

    /* 保存设备 ID */
    err = ef_set_and_save_env("dev_id", dev_id);
    if (err != EF_NO_ERR)
    {
        LOG_E("save device info(dev_id : %s) failed!", dev_id);
        return -RT_ERROR;
    }

    /* 保存设备 api_key */
    err = ef_set_and_save_env("api_key", api_key);
    if (err != EF_NO_ERR)
    {
        LOG_E("save device info(api_key : %s) failed!", api_key);
        return -RT_ERROR;
    }

    /* 保存环境变量：已经注册 */
    err = ef_set_and_save_env("already_register", "1");
    if (err != EF_NO_ERR)
    {
        LOG_E("save already_register failed!");
        return -RT_ERROR;
    }

    return RT_EOK;
}

rt_err_t onenet_port_get_register_info(char *dev_name, char *auth_info)
{
    rt_uint32_t cpuid[2] = {0};
    EfErrCode err = EF_NO_ERR;

    /* 获取 stm32 uid */
    cpuid[0] = *(volatile rt_uint32_t *)(0x1FFF7590);
    cpuid[1] = *(volatile rt_uint32_t *)(0x1FFF7590 + 4);

    /* 设置设备名和鉴权信息 */
    rt_snprintf(dev_name, ONENET_INFO_AUTH_LEN, "%d%d", cpuid[0], cpuid[1]);
    rt_snprintf(auth_info, ONENET_INFO_AUTH_LEN, "%d%d", cpuid[0], cpuid[1]);

    /* 保存设备鉴权信息 */
    err = ef_set_and_save_env("auth_info", auth_info);
    if (err != EF_NO_ERR)
    {
        LOG_E("save auth_info failed!");
        return -RT_ERROR;
    }

    return RT_EOK;
}

rt_err_t onenet_port_get_device_info(char *dev_id, char *api_key, char *auth_info)
{
    char *info = RT_NULL; /* 临时保存从 KV 存储读出的字符串 */

    /* 获取设备 ID */
    info = ef_get_env("dev_id"); /* 读取按键 dev_id 对应的值 */
    if (info == RT_NULL)         /* 未配置设备 ID */
    {
        LOG_E("read dev_id failed!");
        return -RT_ERROR;
    }
    else
    {
        rt_snprintf(dev_id, ONENET_INFO_AUTH_LEN, "%s", info); /* 拷贝到输出缓冲并限定长度 */
    }

    /* 获取 api_key */
    info = ef_get_env("api_key"); /* 读取 api_key 对应的值 */
    if (info == RT_NULL)          /* 未配置 api_key */
    {
        LOG_E("read api_key failed!");
        return -RT_ERROR;
    }
    else
    {
        rt_snprintf(api_key, ONENET_INFO_AUTH_LEN, "%s", info); /* 拷贝到输出缓冲并限定长度 */
    }

    /* 获取设备鉴权信息 */
    info = ef_get_env("auth_info"); /* 读取 auth_info 对应的值 */
    if (info == RT_NULL)            /* 未配置鉴权信息 */
    {
        LOG_E("read auth_info failed!");
        return -RT_ERROR;
    }
    else
    {
        rt_snprintf(auth_info, ONENET_INFO_AUTH_LEN, "%s", info); /* 拷贝到输出缓冲并限定长度 */
    }

    return RT_EOK;
}

rt_bool_t onenet_port_is_registed(void)
{
    char *already_register = RT_NULL;

    /* 检查设备是否已经注册 */
    already_register = ef_get_env("already_register");
    if (already_register == RT_NULL)
    {
        return RT_FALSE;
    }

    return already_register[0] == '1' ? RT_TRUE : RT_FALSE;
}

static void onenet_cmd_rsp_cb(uint8_t *recv_data, size_t recv_size, uint8_t **resp_data, size_t *resp_size)
{
    char res_buf[20] = {0};

    LOG_D("recv data is %.*s", recv_size, recv_data);

    /* 命令匹配 */
    if (rt_strncmp((const char *)recv_data, "ledon", 5) == 0)
    {
        /* 开灯 */
        rt_pin_write(LED_PIN, PIN_LOW);

        rt_snprintf(res_buf, sizeof(res_buf), "led is on");

        LOG_D("led is on");
    }
    else if (rt_strncmp((const char *)recv_data, "ledoff", 6) == 0)
    {
        /* 关灯 */
        rt_pin_write(LED_PIN, PIN_HIGH);

        rt_snprintf(res_buf, sizeof(res_buf), "led is off");

        LOG_D("led is off");
    }

    /* 开发者必须使用 ONENET_MALLOC 为响应数据申请内存 */
    *resp_data = (uint8_t *)ONENET_MALLOC(strlen(res_buf) + 1);

    strncpy((char *)resp_data, res_buf, strlen(res_buf));

    *resp_size = strlen(res_buf);
}
