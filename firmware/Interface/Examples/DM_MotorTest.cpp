///*******************************************************************************
// * Copyright (c) 2026.
// * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
// * All rights reserved.
// ******************************************************************************/
//
//#include "Task.h"
//#include "ServoMotors/DM_Motor.hpp"
//#include "VOFA.hpp"
//
//#define DIRECT_POSITION {Motor_Ctrl_Type_e::Position, Motor_Ctrl_Type_e::Position, true}
//#define DIRECT_TORQUE {Motor_Ctrl_Type_e::Torque, Motor_Ctrl_Type_e::Torque}
//auto manipulatorControllers = CreateControllers<Amplifier<1>, 5>();
//
//// DM_Motor<1> motor1(DIRECT_POSITION, manipulatorControllers[0], 0x01);
//// DM_Motor<1> motor2(DIRECT_POSITION, manipulatorControllers[1], 0x02);
//// DM_Motor<1> motor3(DIRECT_POSITION, manipulatorControllers[2], 0x03);
//// DM_Motor<1> motor4(DIRECT_POSITION, manipulatorControllers[3], 0x04);
//DM_Motor<1> motor5(DIRECT_TORQUE, manipulatorControllers[4], 0x05);
//
//float angle1{};
//float angle2{};
//float angle3{};
//float angle4{};
//
//void DMTask() {
//    VOFA<6>::GetInstance().SetData(motor5.GetState().position,0);
//    VOFA<6>::GetInstance().SetData(motor5.GetState().speed,1);
//    VOFA<6>::GetInstance().SetData(motor5.GetState().torque,2);
//    // motor1.SetTargetAngle(angle1);
//    // motor2.SetTargetAngle(angle2);
//    // motor3.SetTargetAngle(angle3);
//    // motor4.SetTargetAngle(angle4);
//}
//TASK_EXPORT(DMTask);