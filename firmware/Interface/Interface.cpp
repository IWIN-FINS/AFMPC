#include "ProjectConfig.h"
#include "DeviceBase.h"
#include "Task.h"

// TaskSUB.cpp 中定义
void SUBPressure_Setup();
void SUBPressure_Loop();

#ifdef __cplusplus
extern "C" {
#endif

void Setup() {
    SUBPressure_Setup();
}

void Loop() {
    SUBPressure_Loop();
}

#ifdef __cplusplus
}
#endif

void MainRTLoop() {
    HAL_IWDG_Refresh(&hiwdg);
    DeviceBase::DevicesHandle();
    RunAllTasks();
}

/*****  不要修改以下代码 *****/

#ifdef __cplusplus
extern "C" {
#endif

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    if (htim == &TIM_Control) {
        MainRTLoop();
    }
}

#ifdef __cplusplus
}
#endif