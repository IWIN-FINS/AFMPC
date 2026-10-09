/*******************************************************************************
* Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#include "Bus/UART_Base.hpp"

#ifdef __cplusplus
extern "C" {
#endif

volatile uint32_t g_uart3_rx_event_count = 0;
volatile uint32_t g_uart3_rx_byte_count = 0;
volatile uint32_t g_uart3_rx_error_count = 0;

// 发送完成中断回调函数
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart) {
    FineMoteAux_UART<>::OnTxComplete(huart);
}

// 接收中断回调函数
// void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
//     uint16_t receivedSize = huart->RxXferSize;
//     FineMoteAux_UART<>::OnRxComplete(huart, receivedSize);
// }

// 出错中断回调函数
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    if (huart == BSP_UARTList[3]) {
        ++g_uart3_rx_error_count;
    }
    __HAL_UART_CLEAR_OREFLAG(huart);
    FineMoteAux_UART<>::OnRxComplete(huart, 0);
}

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t size) {
    if (huart == BSP_UARTList[3]) {
        ++g_uart3_rx_event_count;
        g_uart3_rx_byte_count += size;
    }
    FineMoteAux_UART<>::OnRxComplete(huart, size);
}

#ifdef __cplusplus
}
#endif
