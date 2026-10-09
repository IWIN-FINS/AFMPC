/*******************************************************************************
* Copyright (c) 2025.
* IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
* All rights reserved.
******************************************************************************/

#ifndef V5_SUB_HPP
#define V5_SUB_HPP

#define PRESSURE_SENSORS_INIT_SAMPLE_TIMES 20
#define MOTOR_COUNT 8
#define PRESSURE_SENSOR_COUNT 4
#define MPC_COMMAND_TIMEOUT_MS 500U
#define MPC_MOTOR_THROTTLE_LIMIT 0.50f
#define MPC_FORWARD_CHANNEL_LIMIT 0.35f
#define MPC_RIGHT_CHANNEL_LIMIT 0.35f
#define MPC_DOWN_CHANNEL_LIMIT 0.35f
#define MPC_YAW_CHANNEL_LIMIT 0.20f
// Keep direct calibration commands inside the physically measured envelope.
// The operator-authorized 50% limit applies only to the final mixed AUTO/MPC
// output; it must not silently broaden single-motor/channel calibration.
#define MPC_CALIBRATION_THROTTLE_LIMIT 0.10f
#define IMU_FEEDBACK_MAX_AGE_MS 150U
#define INVALID_PRESSURE_SENSOR_INDEX PRESSURE_SENSOR_COUNT

#include "FillLight/SUBLAB_FillLight.hpp"
#include "ESC_Motors/ESC_Motor.hpp"
#include "PressureSensor/PressureSensorBase.hpp"
#include "IMUBase.hpp"
#include <array>
#include <cstring>
#include <cmath>
#include "Clamp.hpp"
#include "arm_math.h"
#include "Matrix/matrix.h"
#include "Control/PID.hpp"
#include "Verification/CRC.h"
#include "Bus/UART_Base.hpp"

volatile float ls_depth = 0.0f;
volatile float ls_target = 0.0f;

struct SUB_Motor_t {
   ESC_Motor *motor;
   float lx;
   float ly;
   float lz;
};

struct SUB_Pressure_Sensor_t {
   PressureSensorBase *sensor;
   float lx;
   float ly;
   float lz;
};

class V5_SUB;

class V5_SUBBuilder {
public:
   V5_SUBBuilder() = default;

   V5_SUBBuilder& AddMotor(ESC_Motor* motor, float x, float y, float z) {
       if (motorCount < MOTOR_COUNT) {
           motors[motorCount++] = {motor, x, y, z};
       }
       return *this;
   }

   V5_SUBBuilder& AddPressureSensor(PressureSensorBase* sensor, float x, float y, float z) {
       if (sensorCount < PRESSURE_SENSOR_COUNT) {
           pressureSensors[sensorCount++] = {sensor, x, y, z};
       }
       return *this;
   }

   V5_SUBBuilder& AddIMU(IMUBase* imu_) {
       imu = imu_;
       return *this;
   }

   V5_SUBBuilder& AddController(ControllerBase* rollController_,
                                ControllerBase* pitchController_,
                                ControllerBase* yawController_,
                                ControllerBase* deepthController_) {
       rollController = rollController_;
       pitchController = pitchController_;
       yawController = yawController_;
       deepthController = deepthController_;
       return *this;
   }

   V5_SUB* Build();

   std::array<SUB_Motor_t, MOTOR_COUNT> motors{};
   size_t motorCount = 0;

   std::array<SUB_Pressure_Sensor_t, PRESSURE_SENSOR_COUNT> pressureSensors{};
   size_t sensorCount = 0;

   IMUBase* imu = nullptr;

   ControllerBase* rollController = nullptr;
   ControllerBase* pitchController = nullptr;
   ControllerBase* yawController = nullptr;
   ControllerBase* deepthController = nullptr;
};

class V5_SUB : public DeviceBase {
public:
    bool GetIMUInitFinished() const { return imuInitFinished; }
    bool GetPressureInitFinished() const { return pressureInitFinished; }
    uint16_t GetPressureSensorOffsetSampleTimes() const { return pressureSensorOffsetSampleTimes; }
    uint8_t GetStateValue() const { return static_cast<uint8_t>(state); }
    bool IsArmed() const { return state == SUB_STATE::STABILIZING; }
    bool IsMPCControlActive() const { return mpcControlActive; }
    bool IsMPCYawDirect() const { return mpcControlActive && mpcYawDirect; }
    bool IsMPCFailsafeActive() const { return mpcFailsafeActive; }
    bool IsCalibrationMotorActive() const { return mpcCalibrationMotorActive; }
    bool IsIMUFeedbackFresh() const { return imuFeedbackFresh; }
    float GetCurrentYaw() { return currentAttitude[2][0]; }
    float GetCurrentYawRate() { return currentAngleRate[2][0]; }
    float GetCurrentDepth() const { return currentDeepth; }
    float GetCurrentPressurePa() const { return currentPressurePa; }
    float GetMPCForwardChannel() const { return mpcForwardChannel; }
    float GetMPCRightChannel() const { return mpcRightChannel; }
    float GetMPCDownChannel() const { return mpcDownChannel; }
    float GetMPCYawChannel() const { return mpcYawChannel; }
    const std::array<float, MOTOR_COUNT>& GetAppliedMotorThrottle() const {
        return appliedMotorThrottle;
    }
    std::array<float, MOTOR_COUNT> GetMotorRPM() const {
        std::array<float, MOTOR_COUNT> rpm{};
        for (size_t i = 0; i < MOTOR_COUNT; ++i) {
            if (motors[i].motor != nullptr) {
                rpm[i] = motors[i].motor->GetRPM();
            }
        }
        return rpm;
    }
    uint8_t GetMotorRPMValidMask() const {
        uint8_t mask = 0;
        for (size_t i = 0; i < MOTOR_COUNT; ++i) {
            if (motors[i].motor != nullptr && motors[i].motor->IsRPMValid()) {
                mask |= static_cast<uint8_t>(1u << i);
            }
        }
        return mask;
    }

   explicit V5_SUB(V5_SUBBuilder const &temp) {
       motors = temp.motors;
       pressureSensors = temp.pressureSensors;
       imu = temp.imu;
       rollController = temp.rollController;
       pitchController = temp.pitchController;
       yawController = temp.yawController;
       deepthController = temp.deepthController;

       if (rollController)  rollController->SetFeedback({&currentAngleRate[0][0]});
       if (pitchController) pitchController->SetFeedback({&currentAngleRate[1][0]});
       if (yawController)   yawController->SetFeedback({&currentAngleRate[2][0]});
       if (deepthController) deepthController->SetFeedback({&currentDeepth});

       if (rollController)  rollController->SetTarget(&targetAngleRate[0][0]);
       if (pitchController) pitchController->SetTarget(&targetAngleRate[1][0]);
       if (yawController)   yawController->SetTarget(&targetAngleRate[2][0]);
       if (deepthController) deepthController->SetTarget(&targetDeepth);

       A = Matrixf<4, 3>{
           -1.0f,  1.0f,  1.0f,
            1.0f,  1.0f, -1.0f,
            1.0f, -1.0f,  1.0f,
            1.0f,  1.0f,  1.0f
       };

       B = Matrixf<4, 3>{
           -1.0f, -1.0f, -1.0f,
           -1.0f, -1.0f,  1.0f,
            1.0f, -1.0f,  1.0f,
           -1.0f,  1.0f,  1.0f
       };
   }

   void EnableMotors() {
       for (auto &motor : motors) {
           if (motor.motor != nullptr) {
               motor.motor->EnableMotor();
           }
       }
   }

   void DisableMotors() {
       for (size_t i = 0; i < MOTOR_COUNT; ++i) {
           if (motors[i].motor != nullptr) {
               motors[i].motor->SetThrottle(0.0f);
               motors[i].motor->DisableMotor();
           }
           appliedMotorThrottle[i] = 0.0f;
       }
   }

   void Disarm() {
       SetArmed(false);
   }

   void Command(float yawRate, float FBSpeed, float LRSpeed, float downSpeed, float upSpeed, uint8_t startButton) {
       if (mpcControlActive) {
           targetAttitude[2][0] = currentAttitude[2][0];
       }
       mpcControlActive = false;
       mpcCalibrationMotorActive = false;
       mpcCalibrationChannelActive = false;
       mpcCalibrationRollOnly = false;
       mpcCalibrationPitchOnly = false;
       mpcCalibrationYawOnly = false;
       mpcCalibrationMotorThrottle = 0.0f;
       mpcFailsafeActive = false;
       targetDeepth += (downSpeed - upSpeed) * 0.001f * 0.1f;
       targetDeepth = Clamp(targetDeepth, 0.0f, 15.0f);
       ls_target=targetDeepth;

       targetAttitude[2][0] += yawRate * 0.001f * 0.4f;
       targetVel[0][0] = FBSpeed * 0.35f;
       targetVel[1][0] = LRSpeed * 0.35f;

       SwitchState(startButton);
   }

   bool CommandMPC(float forwardChannel,
                   float rightChannel,
                   float downChannel,
                   float yawChannel,
                   bool yawDirect,
                   bool armed) {
       if (!std::isfinite(forwardChannel) ||
           !std::isfinite(rightChannel) ||
           !std::isfinite(downChannel) ||
           !std::isfinite(yawChannel)) {
           EnterMPCFailsafe();
           return false;
       }

       const bool enteringLocalYaw = !yawDirect && (!mpcControlActive || mpcYawDirect);
       mpcControlActive = true;
       mpcCalibrationMotorActive = false;
       mpcCalibrationChannelActive = false;
       mpcCalibrationRollOnly = false;
       mpcCalibrationPitchOnly = false;
       mpcCalibrationYawOnly = false;
       mpcCalibrationMotorThrottle = 0.0f;
       mpcYawDirect = yawDirect;
       lastMPCCommandTick = HAL_GetTick();

       if (!armed) {
           mpcForwardChannel = 0.0f;
           mpcRightChannel = 0.0f;
           mpcDownChannel = 0.0f;
           mpcYawChannel = 0.0f;
           SetArmed(false);
           mpcRearmRequired = false;
           mpcFailsafeActive = false;
           return true;
       }

       if (mpcRearmRequired) {
           EnterMPCFailsafe();
           return false;
       }

       if (enteringLocalYaw) {
           targetAttitude[2][0] = currentAttitude[2][0];
           // Direct yaw bypasses the rate PID.  Drop the controller state when
           // returning to heading hold so an old integral/output cannot create
           // a kick while the new heading target is captured.
           if (yawController) yawController->Reset();
       }
       mpcFailsafeActive = false;
       mpcForwardChannel = Clamp(
           forwardChannel, -MPC_FORWARD_CHANNEL_LIMIT, MPC_FORWARD_CHANNEL_LIMIT);
       mpcRightChannel = Clamp(
           rightChannel, -MPC_RIGHT_CHANNEL_LIMIT, MPC_RIGHT_CHANNEL_LIMIT);
       mpcDownChannel = Clamp(
           downChannel, -MPC_DOWN_CHANNEL_LIMIT, MPC_DOWN_CHANNEL_LIMIT);
       mpcYawChannel = Clamp(
           yawChannel, -MPC_YAW_CHANNEL_LIMIT, MPC_YAW_CHANNEL_LIMIT);
       SetArmed(true);
       return true;
   }

   bool CommandCalibrationChannels(float forwardChannel,
                                   float rightChannel,
                                   float downChannel,
                                   float yawChannel,
                                   bool armed) {
       if (!std::isfinite(forwardChannel) ||
           !std::isfinite(rightChannel) ||
           !std::isfinite(downChannel) ||
           !std::isfinite(yawChannel) ||
           std::fabs(forwardChannel) > MPC_CALIBRATION_THROTTLE_LIMIT + 1.0e-6f ||
           std::fabs(rightChannel) > MPC_CALIBRATION_THROTTLE_LIMIT + 1.0e-6f ||
           std::fabs(downChannel) > MPC_CALIBRATION_THROTTLE_LIMIT + 1.0e-6f ||
           std::fabs(yawChannel) > MPC_CALIBRATION_THROTTLE_LIMIT + 1.0e-6f) {
           EnterMPCFailsafe();
           return false;
       }
       const bool accepted = CommandMPC(
           forwardChannel,
           rightChannel,
           downChannel,
           yawChannel,
           true,
           armed);
       if (accepted) {
           mpcCalibrationChannelActive = armed;
       }
       return accepted;
   }

   bool CommandCalibrationMotor(uint8_t motorIndex,
                                float throttle,
                                bool armed) {
       if (motorIndex >= MOTOR_COUNT ||
           !std::isfinite(throttle) ||
           std::fabs(throttle) > MPC_CALIBRATION_THROTTLE_LIMIT + 1.0e-6f) {
           EnterMPCFailsafe();
           return false;
       }

       mpcControlActive = true;
       mpcYawDirect = true;
       mpcCalibrationMotorActive = true;
       mpcCalibrationChannelActive = false;
       mpcCalibrationRollOnly = false;
       mpcCalibrationPitchOnly = false;
       mpcCalibrationYawOnly = false;
       lastMPCCommandTick = HAL_GetTick();
       mpcForwardChannel = 0.0f;
       mpcRightChannel = 0.0f;
       mpcDownChannel = 0.0f;
       mpcYawChannel = 0.0f;

       if (!armed) {
           mpcCalibrationMotorThrottle = 0.0f;
           SetArmed(false);
           mpcRearmRequired = false;
           mpcFailsafeActive = false;
           return true;
       }

       if (mpcRearmRequired) {
           EnterMPCFailsafe();
           return false;
       }

       mpcFailsafeActive = false;
       mpcCalibrationMotorIndex = motorIndex;
       mpcCalibrationMotorThrottle = throttle;
       SetArmed(true);
       return true;
   }

   bool CommandCalibrationAttitude(bool rollOnly,
                                   bool pitchOnly,
                                   bool yawOnly,
                                   bool armed) {
       const uint8_t selectedAxisCount = static_cast<uint8_t>(rollOnly) +
                                         static_cast<uint8_t>(pitchOnly) +
                                         static_cast<uint8_t>(yawOnly);
       if (selectedAxisCount != 1U) {
           EnterMPCFailsafe();
           return false;
       }
       const bool continuingSameAxis =
           mpcControlActive &&
           mpcCalibrationRollOnly == rollOnly &&
           mpcCalibrationPitchOnly == pitchOnly &&
           mpcCalibrationYawOnly == yawOnly;
       const bool accepted = CommandMPC(
           0.0f, 0.0f, 0.0f, 0.0f, !yawOnly, armed);
       if (accepted) {
           mpcCalibrationRollOnly = armed && rollOnly;
           mpcCalibrationPitchOnly = armed && pitchOnly;
           mpcCalibrationYawOnly = armed && yawOnly;
           if (armed && !continuingSameAxis) {
               ResetAttitudeControllers();
           }
       }
       return accepted;
   }

   void Handle() final {
       Update();

       if (mpcControlActive && state == SUB_STATE::STABILIZING &&
           static_cast<uint32_t>(HAL_GetTick() - lastMPCCommandTick) > MPC_COMMAND_TIMEOUT_MS) {
           EnterMPCFailsafe();
       }

       switch (state) {
       case SUB_STATE::INIT:
           PressureSensorInit();
           IMUInit();
           if (pressureInitFinished && imuInitFinished) {
               state = SUB_STATE::IDLE;
           }
           break;

       case SUB_STATE::IDLE:
           break;

       case SUB_STATE::STABILIZING:
           if (mpcControlActive && mpcCalibrationMotorActive) {
               ApplyCalibrationMotorThrottle();
               break;
           }
           StabilizeMode();
           if (imuFeedbackFresh) {
               if (!mpcCalibrationPitchOnly && !mpcCalibrationYawOnly &&
                   rollController) rollController->Calc();
               if (!mpcCalibrationRollOnly && !mpcCalibrationYawOnly &&
                   pitchController) pitchController->Calc();
               if (!mpcControlActive || !mpcYawDirect) {
                   if (yawController) yawController->Calc();
               }
           }
           if (!mpcControlActive) {
               if (deepthController) deepthController->Calc();
           }
           MotorPowerDistribution();
           break;

       case SUB_STATE::POSITION_HOLDING:
           PositionHoldMode();
           break;

       default:
           break;
       }
   }

private:
   enum class SUB_STATE {
       INIT = 0X00,
       IDLE = 0X01,
       STABILIZING = 0X02,
       POSITION_HOLDING = 0X03
   } state = SUB_STATE::INIT;

   std::array<SUB_Motor_t, MOTOR_COUNT> motors{};
   std::array<SUB_Pressure_Sensor_t, PRESSURE_SENSOR_COUNT> pressureSensors{};
   IMUBase *imu = nullptr;

   std::array<float, PRESSURE_SENSOR_COUNT> pressureSensorOffset = {0.0f, 0.0f, 0.0f, 0.0f};
   std::array<float, PRESSURE_SENSOR_COUNT> pressureSensorOffsetSum = {0.0f, 0.0f, 0.0f, 0.0f};
   uint16_t pressureSensorOffsetSampleTimes = 0;
   size_t depthPressureSensorIndex = INVALID_PRESSURE_SENSOR_INDEX;

   bool pressureInitFinished = false;
   bool imuInitFinished = false;
   bool imuFeedbackFresh = false;

   // Matrixf's default constructor only binds its CMSIS matrix view; it does
   // not initialize the backing array.  Explicitly zero every controller state
   // so roll/pitch targets cannot inherit arbitrary SRAM contents after reset.
   Matrixf<3, 1> targetAttitude = matrixf::zeros<3, 1>();
   Matrixf<3, 1> targetAngleRate = matrixf::zeros<3, 1>();
   Matrixf<3, 1> currentAngleRate = matrixf::zeros<3, 1>();
   Matrixf<3, 1> currentAttitude = matrixf::zeros<3, 1>();
   Matrixf<3, 1> rollPitchDeepthControllerOutput = matrixf::zeros<3, 1>();
   Matrixf<3, 1> yawFbLrControllerOutput = matrixf::zeros<3, 1>();
   Matrixf<4, 3> A = matrixf::zeros<4, 3>();
   Matrixf<4, 3> B = matrixf::zeros<4, 3>();
   Matrixf<2, 1> targetVel = matrixf::zeros<2, 1>();

   float targetDeepth{0.0f};
   float currentDeepth{0.0f};
   float currentPressurePa{0.0f};
   std::array<float, MOTOR_COUNT> appliedMotorThrottle{};

   bool mpcControlActive = false;
   bool mpcYawDirect = false;
   bool mpcCalibrationMotorActive = false;
   bool mpcCalibrationChannelActive = false;
   bool mpcCalibrationRollOnly = false;
   bool mpcCalibrationPitchOnly = false;
   bool mpcCalibrationYawOnly = false;
   bool mpcFailsafeActive = false;
   bool mpcRearmRequired = true;
   uint32_t lastMPCCommandTick = 0;
   float mpcForwardChannel = 0.0f;
   float mpcRightChannel = 0.0f;
   float mpcDownChannel = 0.0f;
   float mpcYawChannel = 0.0f;
   uint8_t mpcCalibrationMotorIndex = 0;
   float mpcCalibrationMotorThrottle = 0.0f;

   ControllerBase* rollController = nullptr;
   ControllerBase* pitchController = nullptr;
   ControllerBase* yawController = nullptr;
   ControllerBase* deepthController = nullptr;

private:
   void PressureSensorInit() {
       if (pressureInitFinished) {
           return;
       }

       // Channel 3 is known faulty on this vehicle.  On each power-up, lock to
       // the first channel among 0..2 that actually returns a valid absolute
       // pressure instead of letting one unavailable channel block all control.
       if (depthPressureSensorIndex >= PRESSURE_SENSOR_COUNT) {
           for (size_t candidate = 0; candidate < 3U; ++candidate) {
               if (pressureSensors[candidate].sensor == nullptr) {
                   continue;
               }
               const float candidatePressure =
                   pressureSensors[candidate].sensor->GetPressure();
               if (std::isfinite(candidatePressure) &&
                   candidatePressure >= 50000.0f) {
                   depthPressureSensorIndex = candidate;
                   break;
               }
           }
       }
       if (depthPressureSensorIndex >= PRESSURE_SENSOR_COUNT) {
           return;
       }

       const size_t index = depthPressureSensorIndex;
       const float pressure = pressureSensors[index].sensor->GetPressure();
       // MS5837_TCA_Channel::GetPressure() returns absolute pressure in Pa.
       if (!std::isfinite(pressure) || pressure < 50000.0f) {
           return;
       }
       pressureSensorOffsetSum[index] += pressure;
       pressureSensorOffsetSampleTimes++;

       if (pressureSensorOffsetSampleTimes >= PRESSURE_SENSORS_INIT_SAMPLE_TIMES) {
           pressureSensorOffset[index] =
               pressureSensorOffsetSum[index] /
               static_cast<float>(PRESSURE_SENSORS_INIT_SAMPLE_TIMES);

           currentDeepth = 0.0f;
           targetDeepth = 0.0f;
           pressureInitFinished = true;
       }
   }

   void IMUInit() {
       if (imuInitFinished) {
           return;
       }
       if (imu != nullptr && imu->IsDataGet()) {
           imuInitFinished = true;
           // Update() has already converted the IMU convention to body FRD.
           targetAttitude[2][0] = currentAttitude[2][0];
       }
   }

   float SqrtControler(float error, float errorLimit, float accLimit, float p) {
       if (error > errorLimit) {
           error = errorLimit;
       } else if (error < -errorLimit) {
           error = -errorLimit;
       }

       if (errorLimit < 1e-6f) {
           return error * p;
       } else if (p < 1e-6f) {
           if (error > 1e-6f) {
               return sqrtf(2.0f * accLimit * error);
           } else if (error < -1e-6f) {
               return -sqrtf(-2.0f * accLimit * error);
           } else {
               return 0.0f;
           }
       } else {
           float linear_dist = accLimit / (p * p);

           if (error > linear_dist) {
               return sqrtf(2.0f * accLimit * (error - (linear_dist / 2.0f)));
           } else if (error < -linear_dist) {
               return -sqrtf(2.0f * accLimit * (-error - (linear_dist / 2.0f)));
           } else {
               return error * p;
           }
       }
   }

   void Update() {
       if (imu != nullptr) {
           const bool wasFresh = imuFeedbackFresh;
           imuFeedbackFresh = imu->IsFresh(IMU_FEEDBACK_MAX_AGE_MS);
           if (imuFeedbackFresh) {
               const auto imuAngleRate = imu->getAngleRateArray();
               const auto imuEuler = imu->getEulerArray();

               // Real-vehicle checks show that positive raw IMU x/y/z rotation is
               // negative body-FRD roll/pitch/yaw.  Keep both layers of the
               // attitude controller in the same FRD convention as the mixer.
               for (size_t axis = 0; axis < 3; ++axis) {
                   currentAngleRate[axis][0] = -imuAngleRate[axis];
                   currentAttitude[axis][0] = -imuEuler[axis];
               }

               // Do not demand a catch-up yaw rotation when IMU packets resume.
               if (!wasFresh && imuInitFinished) {
                   targetAttitude[2][0] = currentAttitude[2][0];
               }
               Matrixf<3,1> attitudeError = targetAttitude - currentAttitude;
               targetAngleRate[0][0] = SqrtControler(attitudeError[0][0], 2.0f, 1.0f, 2.5f);
               targetAngleRate[1][0] = SqrtControler(attitudeError[1][0], 2.0f, 1.0f, 2.5f);
               targetAngleRate[2][0] = SqrtControler(attitudeError[2][0], 2.0f, 1.0f, 2.5f);
           } else {
               for (size_t axis = 0; axis < 3; ++axis) {
                   targetAngleRate[axis][0] = 0.0f;
               }
               if (rollController) rollController->Reset();
               if (pitchController) pitchController->Reset();
               if (yawController) yawController->Reset();
           }
       }

       const size_t pressureIndex = depthPressureSensorIndex;
       float waterPressure = 0.0f;
       if (pressureIndex < PRESSURE_SENSOR_COUNT &&
           pressureSensors[pressureIndex].sensor != nullptr) {
           waterPressure = pressureSensors[pressureIndex].sensor->GetPressure() -
                           pressureSensorOffset[pressureIndex];
       }
       if (waterPressure < 0.0f) {
           waterPressure = 0.0f;
       }

       currentPressurePa = waterPressure;
       currentDeepth = waterPressure / 9.8f / 1000.0f;
       ls_depth = currentDeepth;
   }

   void MotorPowerDistribution() {
       float rollOutput = mpcCalibrationChannelActive || mpcCalibrationPitchOnly ||
                          mpcCalibrationYawOnly
           ? 0.0f
           : (imuFeedbackFresh
               ? Clamp(1.0f * (rollController ? rollController->GetOutput() : 0.0f), -0.2f, 0.2f)
               : 0.0f);
       float pitchOutput = mpcCalibrationChannelActive || mpcCalibrationRollOnly ||
                           mpcCalibrationYawOnly
           ? 0.0f
           : (imuFeedbackFresh
               ? Clamp(1.0f * (pitchController ? pitchController->GetOutput() : 0.0f), -0.2f, 0.2f)
               : 0.0f);
       float yawOutput = mpcControlActive && mpcYawDirect
           ? mpcYawChannel
           : (imuFeedbackFresh
               ? Clamp(1.0f * (yawController ? yawController->GetOutput() : 0.0f), -0.2f, 0.2f)
               : 0.0f);
       float deepthOutput = mpcControlActive
           ? mpcDownChannel
           : Clamp(1.0f * (deepthController ? deepthController->GetOutput() : 0.0f), -0.5f, 0.5f);

       rollPitchDeepthControllerOutput[0][0] = rollOutput;
       rollPitchDeepthControllerOutput[1][0] = pitchOutput;
       rollPitchDeepthControllerOutput[2][0] = deepthOutput;

       Matrixf<4,1> upperMotorOutPut = A * rollPitchDeepthControllerOutput;
       upperMotorOutPut[0][0] = fabsf(upperMotorOutPut[0][0]) < 0.01f ? 0.0f : upperMotorOutPut[0][0];
       upperMotorOutPut[1][0] = fabsf(upperMotorOutPut[1][0]) < 0.01f ? 0.0f : upperMotorOutPut[1][0];
       upperMotorOutPut[2][0] = fabsf(upperMotorOutPut[2][0]) < 0.01f ? 0.0f : upperMotorOutPut[2][0];
       upperMotorOutPut[3][0] = fabsf(upperMotorOutPut[3][0]) < 0.01f ? 0.0f : upperMotorOutPut[3][0];

       yawFbLrControllerOutput[0][0] = yawOutput;
       yawFbLrControllerOutput[1][0] = mpcControlActive ? mpcForwardChannel : targetVel[0][0];
       yawFbLrControllerOutput[2][0] = mpcControlActive ? mpcRightChannel : targetVel[1][0];

       Matrixf<4,1> lowerMotorOutPut = B * yawFbLrControllerOutput;
       lowerMotorOutPut[0][0] = fabsf(lowerMotorOutPut[0][0]) < 0.01f ? 0.0f : lowerMotorOutPut[0][0];
       lowerMotorOutPut[1][0] = fabsf(lowerMotorOutPut[1][0]) < 0.01f ? 0.0f : lowerMotorOutPut[1][0];
       lowerMotorOutPut[2][0] = fabsf(lowerMotorOutPut[2][0]) < 0.01f ? 0.0f : lowerMotorOutPut[2][0];
       lowerMotorOutPut[3][0] = fabsf(lowerMotorOutPut[3][0]) < 0.01f ? 0.0f : lowerMotorOutPut[3][0];

       if (mpcControlActive) {
           // Translational and attitude demands share each thruster.  Scale
           // the complete mixed vector uniformly so the final motor limit is
           // respected without changing the requested wrench direction.
           float maximumThrottle = 0.0f;
           for (size_t motor = 0; motor < 4; ++motor) {
               const float upperMagnitude = fabsf(upperMotorOutPut[motor][0]);
               const float lowerMagnitude = fabsf(lowerMotorOutPut[motor][0]);
               if (upperMagnitude > maximumThrottle) {
                   maximumThrottle = upperMagnitude;
               }
               if (lowerMagnitude > maximumThrottle) {
                   maximumThrottle = lowerMagnitude;
               }
           }
           if (maximumThrottle > MPC_MOTOR_THROTTLE_LIMIT) {
               const float scale = MPC_MOTOR_THROTTLE_LIMIT / maximumThrottle;
               for (size_t motor = 0; motor < 4; ++motor) {
                   upperMotorOutPut[motor][0] *= scale;
                   lowerMotorOutPut[motor][0] *= scale;
               }
           }
       }

       ApplyMotorThrottle(2, upperMotorOutPut[0][0]);
       ApplyMotorThrottle(3, upperMotorOutPut[1][0]);
       ApplyMotorThrottle(4, upperMotorOutPut[2][0]);
       ApplyMotorThrottle(7, upperMotorOutPut[3][0]);

       ApplyMotorThrottle(0, lowerMotorOutPut[0][0]);
       ApplyMotorThrottle(1, lowerMotorOutPut[1][0]);
       ApplyMotorThrottle(5, lowerMotorOutPut[2][0]);
       ApplyMotorThrottle(6, lowerMotorOutPut[3][0]);
   }

   void ApplyMotorThrottle(size_t index, float throttle) {
       if (index >= MOTOR_COUNT) {
           return;
       }
       throttle = Clamp(throttle, -1.0f, 1.0f);
       if (mpcControlActive) {
           throttle = Clamp(
               throttle,
               -MPC_MOTOR_THROTTLE_LIMIT,
               MPC_MOTOR_THROTTLE_LIMIT);
       }
       appliedMotorThrottle[index] = throttle;
       if (motors[index].motor != nullptr) {
           motors[index].motor->SetThrottle(throttle);
       }
   }

   void ApplyCalibrationMotorThrottle() {
       for (size_t index = 0; index < MOTOR_COUNT; ++index) {
           ApplyMotorThrottle(
               index,
               index == mpcCalibrationMotorIndex
                   ? mpcCalibrationMotorThrottle
                   : 0.0f);
       }
   }

   void StabilizeMode() {
   }

   void PositionHoldMode() {
   }

   void ResetAttitudeControllers() {
       for (size_t axis = 0; axis < 3; ++axis) {
           targetAngleRate[axis][0] = 0.0f;
       }
       if (rollController) rollController->Reset();
       if (pitchController) pitchController->Reset();
       if (yawController) yawController->Reset();
   }

   void EnterMPCFailsafe() {
       mpcForwardChannel = 0.0f;
       mpcRightChannel = 0.0f;
       mpcDownChannel = 0.0f;
       mpcYawChannel = 0.0f;
       mpcCalibrationMotorActive = false;
       mpcCalibrationChannelActive = false;
       mpcCalibrationRollOnly = false;
       mpcCalibrationPitchOnly = false;
       mpcCalibrationYawOnly = false;
       mpcCalibrationMotorThrottle = 0.0f;
       mpcFailsafeActive = true;
       mpcRearmRequired = true;
       SetArmed(false);
   }

   void SetArmed(bool armed) {
       if (armed) {
           if (state == SUB_STATE::IDLE) {
               ResetAttitudeControllers();
               EnableMotors();
               targetAttitude[2][0] = currentAttitude[2][0];
               state = SUB_STATE::STABILIZING;
           }
           return;
       }

       ResetAttitudeControllers();
       if (state == SUB_STATE::STABILIZING || state == SUB_STATE::POSITION_HOLDING) {
           DisableMotors();
           state = SUB_STATE::IDLE;
       }
   }

   void SwitchState(uint8_t buttonState) {
       static uint8_t lastButtonState = 0;
       if (buttonState != lastButtonState && buttonState == 1) {
           if (state == SUB_STATE::IDLE) {
               ResetAttitudeControllers();
               EnableMotors();
               targetAttitude[2][0] = currentAttitude[2][0];
               state = SUB_STATE::STABILIZING;
           } else if (state == SUB_STATE::STABILIZING) {
               ResetAttitudeControllers();
               DisableMotors();
               state = SUB_STATE::IDLE;
           }
       }
       lastButtonState = buttonState;
   }
};

inline V5_SUB* V5_SUBBuilder::Build() {
   return new V5_SUB(*this);
}

#endif // V5_SUB_HPP
