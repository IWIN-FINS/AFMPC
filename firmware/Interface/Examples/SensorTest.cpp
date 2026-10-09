// /*******************************************************************************
//  * Copyright (c) 2026.
//  * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
//  * All rights reserved.
//  ******************************************************************************/
//
// #include "Task.h"
// #include "PressureSensor/XS_PressureSensor.hpp"
// #include "VOFA.hpp"
//
// XS_PressureSensor<8> pressureSensor1(0x01);
// XS_PressureSensor<8> pressureSensor2(0x02);
// XS_PressureSensor<8> pressureSensor3(0x03);
// XS_PressureSensor<8> pressureSensor4(0x04);
//
// void SensorTask() {
//     VOFA<3>::GetInstance().SetData(pressureSensor1.GetPressure(),0);
//     VOFA<3>::GetInstance().SetData(pressureSensor2.GetPressure(),1);
//     VOFA<3>::GetInstance().SetData(pressureSensor3.GetPressure(),2);
//     VOFA<3>::GetInstance().SetData(pressureSensor4.GetPressure(),3);
//
//
//     VOFA<3>::GetInstance().SetData(pressureSensor1.GetFrequency(),4);
//     VOFA<3>::GetInstance().SetData(pressureSensor2.GetFrequency(),5);
//     VOFA<3>::GetInstance().SetData(pressureSensor3.GetFrequency(),6);
//     VOFA<3>::GetInstance().SetData(pressureSensor4.GetFrequency(),7);
//     VOFA<3>::GetInstance().SetData(static_cast<float>(XS_PressureSensor<8>::GetInstanceNum()),8);
// }
// TASK_EXPORT(SensorTask);