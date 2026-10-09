/*******************************************************************************
* Copyright (c) 2025.
* IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
* All rights reserved.
*******************************************************************************/

#include "Bus/DSHOT.hpp"
#include "Bus/UART_Base.hpp"
#include "DSHOT_Motor.hpp"
#include "IMU/H30_IMU.hpp"
#include "SUB/V5_SUB.hpp"
#include "SUBLAB_FillLight.hpp"
#include "Task.h"
#include "V5Streamer/Streamer.hpp"
#include "ServoMotors/ServoS30.hpp"
#include "PressureSensor/MS5837_TCA_Array.hpp"
#include "PressureSensor/MS5837_TCA_Channel.hpp"

#ifndef RAD_TO_DEG_F
#define RAD_TO_DEG_F 57.2957795f
#endif

#define M6 7//1号电机
#define M2 3//2号电机
#define M3 1//3号电机
#define M4 2//4号电机
#define M5 8//5号电机
#define M1 4//6号电机
#define M7 5 //7号电机
#define M8 6//8号电机

#define L1 11
#define L2 12

ServoS30<9> camPitchServo;       // PWM ID 9
ServoS30<10> camYawServo;    // PWM ID 10
float camYawAngle{0};
float camPitchAngle{0};

ESC_Motor* motorLFLower = &DSHOT_Motor<M1>::GetInstance();
ESC_Motor* motorLFUpper = &DSHOT_Motor<M2>::GetInstance();
ESC_Motor* motorLBUpper = &DSHOT_Motor<M3>::GetInstance();
ESC_Motor* motorLBLower = &DSHOT_Motor<M4>::GetInstance();
ESC_Motor* motorRBLower = &DSHOT_Motor<M5>::GetInstance();
ESC_Motor* motorRBUpper = &DSHOT_Motor<M6>::GetInstance();
ESC_Motor* motorRFUpper = &DSHOT_Motor<M7>::GetInstance();
ESC_Motor* motorRFLower = &DSHOT_Motor<M8>::GetInstance();

// =========================
// 4 路 TCA + MS5837 实例
// =========================
static MS5837_TCA_Array::Config g_ms5837_cfg{
    .hi2c = &hi2c2,
    .tca_addr7 = 0x70,
    .sensor_count = 4,
    .channels = {0, 1, 2, 3},
    .zero_cal_samples = 64,
    .filter_window = 8,
    .sensor_addr_primary = 0x76,
    .sensor_addr_secondary = 0x77,
    .i2c_timeout_ms = 100,
    .convert_delay_ms = 10
};

static MS5837_TCA_Array g_ms5837_array(g_ms5837_cfg);

static MS5837_TCA_Channel g_ms5837_ch0(g_ms5837_array, 0);
static MS5837_TCA_Channel g_ms5837_ch1(g_ms5837_array, 1);
static MS5837_TCA_Channel g_ms5837_ch2(g_ms5837_array, 2);
static MS5837_TCA_Channel g_ms5837_ch3(g_ms5837_array, 3);

// Setup/Loop hook 状态
static bool g_ms5837_init_done = false;

// =========================
// 给 Interface.cpp 调用的 hook
// =========================
void SUBPressure_Setup() {
    g_ms5837_init_done = g_ms5837_array.Init();
}

void SUBPressure_Loop() {
    if (!g_ms5837_init_done) return;
    g_ms5837_array.Poll();
}

/*** 上位机数据 ***/
Streamer v5streamer;
UARTBuffer<3, 64> uart3Buffer([](uint8_t* data, size_t length) {
    v5streamer.Decode(data, length);
});

uint8_t data_t[9]{0X59 , 0X53 , 0X4D , 0X12 , 0X00 , 0X02 , 0X02 , 0X63, 0XCF};

/*** IMU LinkScope调试变量 ***/

volatile float ls_start_button = 0.0f;
volatile float ls_imu_init_finished = 0.0f;
volatile float ls_pressure_init_finished = 0.0f;
volatile float ls_pressure_offset_sample_times = 0.0f;
volatile float ls_state = 0.0f;

volatile float ls_imu_online = 0.0f;      // 1=在线，0=离线
volatile float ls_imu_rx_cnt = 0.0f;      // UART7接收回调累计次数
volatile float ls_imu_rx_dt_ms = 0.0f;    // 距离上次收到IMU数据的时间(ms)

/*** IMU数据 ***/
H30_IMU imu;
UARTBuffer<7, 100> uart7Buffer([](uint8_t* data, size_t length) {
    imu.Decode(data, length);
});

/*** 控制器 ***/
constexpr PID_Param_t rollRatePID = {0.10f, 0.0001f, 0.0f, 100, 2000};
auto rollRateController = CreateControllers<PID, 1>(rollRatePID);

// The water test with the corrected pitch mixer restored the mean attitude,
// but the original 1 kHz gains produced a visible 0.45 Hz oscillation and
// repeatedly drove the mixed motors to their 10% MPC limit.  Keep a small
// integral term for static trim while reducing the rate-loop aggressiveness.
constexpr PID_Param_t pitchRatePID = {0.10f, 0.0001f, 0.0f, 100, 2000};
auto pitchRateController = CreateControllers<PID, 1>(pitchRatePID);

// Repeated +/-5% direct-yaw pulse tests selected this rate-loop compromise:
// The combined sway/AUTO test showed visible yaw oscillation when the lateral
// mixer crossed the per-motor deadband.  Reduce only rate Kp for the next A/B
// test; keep the outer loop and the small bias-rejection integral unchanged.
constexpr PID_Param_t yawRatePID = {0.12f, 0.00005f, 0.0f, 100, 2000};
auto yawRateController = CreateControllers<PID, 1>(yawRatePID);

constexpr PID_Param_t deepthPID = {0.8f, 0.001f, 0.0f, 100, 0.5};
auto deepthController = CreateControllers<PID, 1>(deepthPID);

// =========================
// 用 4 路 MS5837 替换原 XS_PressureSensor
// V5_SUB locks to the first valid absolute-pressure source among channels
// 0..2 on each boot. Channel 3 remains present for hardware diagnostics but is
// never selected because it is faulty on this vehicle. Keep the ordering fixed.
// =========================
V5_SUB* sub = V5_SUBBuilder().AddMotor(motorLFLower, 0.0f, 0.0f, 0.0f).
              AddMotor(motorLFUpper, 1.0f, 1.0f, 0.0f).
              AddMotor(motorLBUpper, -1.0f, 1.0f, 0.0f).
              AddMotor(motorLBLower, 0.0f, 0.0f, 0.0f).
              AddMotor(motorRBLower, 0.0f, 0.0f, 0.0f).
              AddMotor(motorRBUpper, -1.0f, -1.0f, 0.0f).
              AddMotor(motorRFUpper, 1.0f, -1.0f, 0.0f).
              AddMotor(motorRFLower, 0.0f, 0.0f, 0.0f).
              AddPressureSensor(&g_ms5837_ch0, 0.0f, 0.0f, 0.0f).
              AddPressureSensor(&g_ms5837_ch1, 0.0f, 0.0f, 0.0f).
              AddPressureSensor(&g_ms5837_ch2, 0.0f, 0.0f, 0.0f).
              AddPressureSensor(&g_ms5837_ch3, 0.0f, 0.0f, 0.0f).
              AddIMU(&imu).
              AddController(&rollRateController[0], &pitchRateController[0], &yawRateController[0], &deepthController[0]).Build();

void TaskTest() {

    static bool servoInit = false;
    if (!servoInit) {
        PWM_Base<9>::GetInstance().SetFrequency(50);
        PWM_Base<10>::GetInstance().SetFrequency(50);
        camYawServo.SetAngle(135.0f);
        camPitchServo.SetAngle(135.0f);
        servoInit = true;
    }

    // 摇杆转角速度系数，可适当再调小一点
    constexpr float kYawStep   = 0.005f;
    constexpr float kPitchStep = 0.005f;

    // 摇杆死区，防止回中附近抖动
    constexpr int kDeadband = 1;

    // 舵机中心角
    constexpr float kYawCenter   = 135.0f;
    constexpr float kPitchCenter = 150.0f;

    // 读取摇杆值
    int yawInput   = v5streamer.joystickData.camYaw;
    int pitchInput = v5streamer.joystickData.camPitch;

    // 死区处理
    if (std::abs(yawInput) < kDeadband) {
        yawInput = 0;
    }
    if (std::abs(pitchInput) < kDeadband) {
        pitchInput = 0;
    }

    // 累加角度
    camYawAngle   += static_cast<float>(yawInput) * kYawStep;
    camPitchAngle += static_cast<float>(pitchInput) * kPitchStep;

    // 限幅
    Clamp(camYawAngle, -40.f, 40.f);
    Clamp(camPitchAngle, -30.f, 30.f);

    // 输出到舵机
    camYawServo.SetAngle(kYawCenter - camYawAngle);

    // 把这里改成 + ，修正上下颠倒
    camPitchServo.SetAngle(kPitchCenter + camPitchAngle);

    if (v5streamer.IsMPCControlActive()) {
        Streamer::MPCControlData command{};
        if (v5streamer.ConsumeMPCCommand(command)) {
            const bool applied = command.calibrationMotor
                ? sub->CommandCalibrationMotor(command.calibrationMotorIndex,
                                               command.calibrationMotorThrottle,
                                               command.armed)
                : command.calibrationRollOnly || command.calibrationPitchOnly ||
                  command.calibrationYawOnly
                    ? sub->CommandCalibrationAttitude(command.calibrationRollOnly,
                                                      command.calibrationPitchOnly,
                                                      command.calibrationYawOnly,
                                                      command.armed)
                : command.calibrationChannel
                    ? sub->CommandCalibrationChannels(command.forward,
                                                      command.right,
                                                      command.down,
                                                      command.yaw,
                                                      command.armed)
                    : sub->CommandMPC(command.forward,
                                      command.right,
                                      command.down,
                                      command.yaw,
                                      command.yawDirect,
                                      command.armed);
            if (!applied) {
                v5streamer.RecordActuationRejection(
                    Streamer::COMMAND_REJECT_SESSION_REQUIRES_DISARM);
            }
        }
    } else {
        sub->Command(v5streamer.joystickData.yawRate,
                     v5streamer.joystickData.forwardSpeed,
                     v5streamer.joystickData.leftRightSpeed,
                     v5streamer.joystickData.downSpeed,
                     v5streamer.joystickData.upSpeed,
                     v5streamer.joystickData.startButton);
    }

    // 20 Hz v4 feedback.  It links the last command by session/sequence/CRC,
    // reports its accept/reject decision, and includes both the eight physical
    // motor set-points and DSHOT RPM measurements.  The upper controller
    // reconstructs tau_achieved_previous from the applied motor set-points.
    static uint8_t telemetryDivider = 0;
    static uint16_t telemetrySequence = 0;
    static uint8_t telemetryBuffer[192]{};
    static Streamer::TelemetryData telemetry{};
    if (++telemetryDivider >= 50) {
        telemetryDivider = 0;
        telemetry.sequence = telemetrySequence++;
        telemetry.tickMs = HAL_GetTick();
        telemetry.state = sub->GetStateValue();
        telemetry.armed = sub->IsArmed();
        telemetry.mpcDirect = sub->IsMPCControlActive();
        telemetry.yawDirect = sub->IsMPCYawDirect();
        telemetry.failsafe = sub->IsMPCFailsafeActive();
        telemetry.executionFeedbackValid = true;
        telemetry.rpmAvailable = true;
        telemetry.rpmValidMask = sub->GetMotorRPMValidMask();

        const auto quat = imu.getQuatArray();
        const auto angularVelocity = imu.getAngleRateArray();
        const auto linearAcceleration = imu.getAccelArray();
        for (size_t i = 0; i < 4; ++i) {
            telemetry.quatWxyz[i] = quat[i];
        }
        for (size_t i = 0; i < 3; ++i) {
            telemetry.angularVelocityXyz[i] = angularVelocity[i];
            telemetry.linearAccelerationXyz[i] = linearAcceleration[i];
        }
        telemetry.yawRad = sub->GetCurrentYaw();
        telemetry.depthM = sub->GetCurrentDepth();
        telemetry.pressurePa = sub->GetCurrentPressurePa();

        v5streamer.PopulateCommandDiagnostics(telemetry);
        const auto& appliedMotorThrottle = sub->GetAppliedMotorThrottle();
        const auto motorRpm = sub->GetMotorRPM();
        for (size_t i = 0; i < Streamer::THRUSTER_COUNT; ++i) {
            telemetry.appliedMotorThrottle[i] = appliedMotorThrottle[i];
            telemetry.motorRpm[i] = motorRpm[i];
        }
        const size_t telemetryLength = Streamer::EncodeTelemetry(
            telemetry, telemetryBuffer, sizeof(telemetryBuffer));
        if (telemetryLength > 0) {
            UART_Base<3>::GetInstance().Transmit(
                telemetryBuffer, static_cast<uint16_t>(telemetryLength));
        }
    }

    ls_start_button = (float)v5streamer.joystickData.startButton;
    ls_imu_init_finished = sub->GetIMUInitFinished() ? 1.0f : 0.0f;
    ls_pressure_init_finished = sub->GetPressureInitFinished() ? 1.0f : 0.0f;
    ls_pressure_offset_sample_times = (float)sub->GetPressureSensorOffsetSampleTimes();
    ls_state = (float)sub->GetStateValue();

    static uint8_t lastState = 0;
    if (v5streamer.joystickData.camFillLight) {
    static bool camLightState = false;
    if(lastState != v5streamer.joystickData.camFillLight) {
        camLightState = !camLightState;
    }
    SUBLAB_FillLight<L1>::GetInstance().SetBrightness(camLightState ? 0.5f : 0.0f);
    SUBLAB_FillLight<L2>::GetInstance().SetBrightness(camLightState ? 0.5f : 0.0f);
    }
    lastState = v5streamer.joystickData.camFillLight;

/*    static uint32_t _initTick = 0;
    _initTick++;
    if (_initTick > 5000) {
        motorLFLower->EnableMotor();
        motorLFUpper->EnableMotor();
        motorRFLower->EnableMotor();
        motorRFUpper->EnableMotor();
        motorLBLower->EnableMotor();
        motorRBUpper->EnableMotor();
        motorLBUpper->EnableMotor();
        motorRBLower->EnableMotor();
        motorLFLower->SetThrottle(0.1f);
        motorRFLower->SetThrottle(0.1f);
        motorRBUpper->SetThrottle(0.1f);
        motorRFUpper->SetThrottle(0.1f);
        motorLBUpper->SetThrottle(0.1f);
        motorLBLower->SetThrottle(0.1f);
        motorLFUpper->SetThrottle(0.1f);
        motorRBLower->SetThrottle(0.1f);
    }*/

/*    static uint8_t lastSwitchState = 0;
    if (v5streamer.joystickData.frameFillLight) {
        static bool switchState = false;
        if (lastSwitchState != v5streamer.joystickData.frameFillLight) {
            switchState = !switchState;
        }
        HAL_GPIO_WritePin(GPIOH, GPIO_PIN_10, (switchState ? GPIO_PIN_SET : GPIO_PIN_RESET));
    }
    lastSwitchState = v5streamer.joystickData.frameFillLight;*/
}
TASK_EXPORT(TaskTest);
