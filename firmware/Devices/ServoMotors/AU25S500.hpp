/*******************************************************************************
 * Copyright (c) 2024.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#ifndef FINEMOTE_AU25S500_H
#define FINEMOTE_AU25S500_H

#include "Bus/RS485_Base.hpp"
#include "Verification/CRC.h"


template<uint8_t BusID>
class AU25S500 : public DeviceBase {
public:
    AU25S500(uint8_t addr, float offset) : commuAgent(addr, [this](uint8_t *data, size_t size) {
        // RS485回调：解析舵机返回的位置反馈数据
        // 返回协议：[0x12][0x4C][0x0A][length][ID][angle_L][angle_H][checksum]
        if (size >= 7 && data[2] == 0x0A) {
            uint16_t rxAngle = data[5] | (data[6] << 8);
            if (rxAngle > 32768) {
                currentAngle = (static_cast<int16_t>(rxAngle) / 10.0f);
            } else {
                currentAngle = (rxAngle / 10.0f);
            }
        }
    }), id(addr), angleOffset(offset){
        this->SetDivisionFactor(50);
    }

    void Handle() override {
        MessageGenerate();
    }

    void SetTargetAngle(float angle) {
        float tmp = angleOffset + angle;
        Clamp(tmp, -135.f, 135.f);
        targetAngle = tmp;
    }

    float GetCurrentAngle() const {
        return currentAngle;
    }

    void SetCurrentAngle(float angle) {
        currentAngle = angle;
    }

private:
    float targetAngle{};
    float currentAngle{};
    float angleOffset{};
    uint8_t txbuf[18] = {};

    void MessageGenerate() {
		//控制舵机位置
        uint16_t txAngle = targetAngle >= 0 ? static_cast<uint16_t>(targetAngle * 10) : static_cast<uint16_t>(65536 + targetAngle * 10);
		uint16_t time = 100; //旋转时间，单位为ms
		uint16_t power = 0; //执行功率
		//调整舵机位置
        txbuf[0] = 0x12;
        txbuf[1] = 0x4C; //帧头
        txbuf[2] = 0x08; //控制命令id
        txbuf[3] = 0x07; //数据包长度
        txbuf[4] = id;
        txbuf[5] = txAngle & 0xFF;        // Low byte
        txbuf[6] = (txAngle >> 8) & 0xFF; // High byte
        txbuf[7] = time;
        txbuf[8] = time>>8u;
		txbuf[9] = power;
        txbuf[10] = power>>8u;
		txbuf[11] = ADD8Calc(txbuf, 11);
        commuAgent.Transmit(txbuf, 12);

		//查询舵机位置
		 txbuf[12] = 0x12;
		 txbuf[13] = 0x4C; //帧头
		 txbuf[14] = 0x0A; //控制命令id
		 txbuf[15] = 0x01; //数据包长度
		 txbuf[16] = id;
		 txbuf[17] = ADD8Calc(txbuf+12, 5);

        //修改舵机ID
        uint8_t targetID = 0x02;
        txbuf[0] = 0x12;
        txbuf[1] = 0x4C; //帧头
        txbuf[2] = 0x04; //控制命令id
        txbuf[3] = 0x03; //数据包长度
        txbuf[4] = id;
        txbuf[5] = 0x22;
        txbuf[6] = targetID;
        txbuf[7] = ADD8Calc(txbuf, 7);
        commuAgent.Transmit(txbuf, 8);
    }

    RS485_Agent<BusID> commuAgent;
    const uint8_t id;
};
#endif
