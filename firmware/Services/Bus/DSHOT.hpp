#ifndef FINEMOTE_DSHOT_H
#define FINEMOTE_DSHOT_H

#define DSHOT_MIN_THROTTLE              (48)
#define DSHOT_MAX_THROTTLE              (2047)
#define DSHOT_3D_FORWARD_MIN_THROTTLE   (1048)
#define DSHOT_3D_BACKWARD_MAX_THROTTLE  (1047)
#define DSHOT_RANGE                     (1999)
#define DSHOT_PACK_HEAD_LENGTH (2)
#define DSHOT_PACK_TAIL_LENGTH (2)
#define DSHOT_FRAME_LENGTH (16)
#define DSHOT_PACK_LENGTH (DSHOT_FRAME_LENGTH + DSHOT_PACK_HEAD_LENGTH + DSHOT_PACK_TAIL_LENGTH)
#define DSHOT_RX_LENGTH (100)
#define MOTOR_POLE_PAIR  8 // 大洋科技 P75 电机极对数


#include "BSP_DSHOT.h"
#include "DeviceBase.h"
#include "etl/map.h"
#include "etl/vector.h"
#include "arm_math.h"


class DSHOT_Port : public DeviceBase {
private:
    struct DSHOT_GPIO_PortPins {
        GPIO_TypeDef *port;
        etl::map<uint16_t, uint16_t, 16> pins; // 使用map存储每个引脚的DshotPack
        DSHOT_GPIO_PortPins(GPIO_TypeDef *_port, uint16_t ID) : port(_port) {
            pins[ID] = 0;
        }
    };
    etl::vector<DSHOT_GPIO_PortPins, 4> gpioPortsQueue = {};
    float RPMs[DSHOT_PIN_NUMBER + 1]{};
    bool RPMValid[DSHOT_PIN_NUMBER + 1]{};
    uint16_t rxBuffer[DSHOT_RX_LENGTH] {};
    uint32_t outPutData[DSHOT_PACK_LENGTH * 3] {};
    uint16_t currentIndex = 0;
    bool isInverted = true; // 是否反相传输，开启时启动接收与解码功能
    bool pin3DMode[DSHOT_PIN_NUMBER + 1]{};

    DSHOT_Port(){}

    void UpdateMiddleBit(etl::map<uint16_t, uint16_t, 16> &pins)
    {
        for (int symbol_index = DSHOT_PACK_HEAD_LENGTH; symbol_index < DSHOT_FRAME_LENGTH + DSHOT_PACK_HEAD_LENGTH; symbol_index++) {
            outPutData[symbol_index * 3 + 1] = 0;          // Reset bits
        }
        for(auto &pin : pins) {
            uint16_t value = pin.second;
            uint16_t gpioPinMusk = BSP_DHOSTPinList[pin.first];
            uint32_t middleBitHigh = isInverted ? gpioPinMusk << 16 : gpioPinMusk;
            uint32_t middleBitLow = isInverted ? gpioPinMusk : gpioPinMusk << 16;
            for (int pos = DSHOT_PACK_HEAD_LENGTH; pos < DSHOT_FRAME_LENGTH + DSHOT_PACK_HEAD_LENGTH; pos++) {
                if (value & 0x8000) {
                    outPutData[pos * 3 + 1] |= middleBitHigh;
                }else {
                    outPutData[pos * 3 + 1] |= middleBitLow;
                }
                value <<= 1;
            }
        }
    }

    void GenerateMessage(etl::map<uint16_t, uint16_t, 16> &pins)
    {
        uint32_t portMask = 0;
        for(auto &pin : pins) {
            portMask |= BSP_DHOSTPinList[pin.first];
        }
        uint32_t resetMask;
        uint32_t setMask;

        if (isInverted) {
            resetMask = portMask;
            setMask = (portMask << 16);
        } else {
            resetMask = (portMask << 16);
            setMask = portMask;
        }

        for(int hold_bit_index = 0; hold_bit_index < DSHOT_PACK_HEAD_LENGTH; hold_bit_index++) {
            outPutData[hold_bit_index * 3 + 0] = resetMask; // Always reset all ports
            outPutData[hold_bit_index * 3 + 1] = resetMask;
            outPutData[hold_bit_index * 3 + 2] = resetMask;
        }
        for (int symbol_index = DSHOT_PACK_HEAD_LENGTH; symbol_index < DSHOT_FRAME_LENGTH + DSHOT_PACK_HEAD_LENGTH; symbol_index++) {
            outPutData[symbol_index * 3 + 0] = setMask ; // Always set all ports
            outPutData[symbol_index * 3 + 1] = 0;          // Reset bits are port dependent
            outPutData[symbol_index * 3 + 2] = resetMask; // Always reset all ports
        }
        for(int hold_bit_index = DSHOT_FRAME_LENGTH + DSHOT_PACK_HEAD_LENGTH; hold_bit_index < DSHOT_PACK_LENGTH; hold_bit_index++) {
            outPutData[hold_bit_index * 3 + 0] = resetMask; // Always reset all ports
            outPutData[hold_bit_index * 3 + 1] = resetMask;
            outPutData[hold_bit_index * 3 + 2] = resetMask;
        }

        UpdateMiddleBit(pins);
    }

    static constexpr uint32_t DSHOT_TELEMETRY_NOEDGE = 0xFFFEu;
    static constexpr uint32_t DSHOT_TELEMETRY_INVALID = 0xFFFFu;
    static constexpr size_t MIN_VALID_BBSAMPLES = 57;
    static constexpr size_t MAX_VALID_BBSAMPLES = 69;

    // Betaflight-compatible bidirectional DSHOT GCR and checksum decoder.
    static uint32_t DecodeGcrValue(uint32_t value) {
        value &= 0xFFFFFu;
        static const uint32_t gcrDecode[32] = {
            0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
            0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu,
            0xFFFFFFFFu, 9u, 10u, 11u, 0xFFFFFFFFu, 13u, 14u, 15u,
            0xFFFFFFFFu, 0xFFFFFFFFu, 2u, 3u, 0xFFFFFFFFu, 5u, 6u, 7u,
            0xFFFFFFFFu, 0u, 8u, 1u, 0xFFFFFFFFu, 4u, 12u, 0xFFFFFFFFu
        };
        uint32_t decoded = gcrDecode[value & 0x1Fu];
        decoded |= gcrDecode[(value >> 5u) & 0x1Fu] << 4u;
        decoded |= gcrDecode[(value >> 10u) & 0x1Fu] << 8u;
        decoded |= gcrDecode[(value >> 15u) & 0x1Fu] << 12u;
        uint32_t checksum = decoded ^ (decoded >> 8u);
        checksum ^= checksum >> 4u;
        if ((checksum & 0xFu) != 0xFu || decoded > 0xFFFFu) {
            return DSHOT_TELEMETRY_INVALID;
        }
        return decoded >> 4u;
    }

    static uint32_t DecodeTelemetryBits(const uint16_t* buffer,
                                        size_t count,
                                        uint16_t pinMask) {
        uint32_t value = 0;
        size_t index = 0;
        while (index + 4 <= count) {
            if (!(buffer[index] & pinMask) || !(buffer[index + 1] & pinMask) ||
                !(buffer[index + 2] & pinMask) || !(buffer[index + 3] & pinMask)) {
                break;
            }
            index += 4;
        }
        while (index < count && (buffer[index] & pinMask)) {
            ++index;
        }
        if (index >= count - MIN_VALID_BBSAMPLES) {
            return DSHOT_TELEMETRY_NOEDGE;
        }

        const size_t remaining = (count - index) < MAX_VALID_BBSAMPLES
            ? count - index
            : MAX_VALID_BBSAMPLES;
        const size_t endIndex = index + remaining;
        size_t oldIndex = index;
        uint16_t lastValue = 0;
        uint32_t bits = 0;
        while (index < endIndex) {
            while (index + 4 <= endIndex) {
                if ((buffer[index] & pinMask) != lastValue ||
                    (buffer[index + 1] & pinMask) != lastValue ||
                    (buffer[index + 2] & pinMask) != lastValue ||
                    (buffer[index + 3] & pinMask) != lastValue) {
                    break;
                }
                index += 4;
            }
            while (index < endIndex && (buffer[index] & pinMask) == lastValue) {
                ++index;
            }
            if (index >= endIndex) {
                break;
            }
            const uint32_t sampleCount = index - oldIndex;
            uint32_t length = (sampleCount + 1u) / 3u;
            if (length < 1u) {
                length = 1u;
            }
            bits += length;
            value <<= length;
            value |= 1u << (length - 1u);
            oldIndex = index;
            lastValue = buffer[index] & pinMask;
        }
        if (bits < 18u || bits > 21u) {
            return DSHOT_TELEMETRY_NOEDGE;
        }
        const uint32_t remainingBits = 21u - bits;
        if (remainingBits > 0u) {
            value <<= remainingBits;
            value |= 1u << (remainingBits - 1u);
        }
        return DecodeGcrValue(value);
    }

    static float DecodeRPM(const uint16_t* buffer, size_t count, uint16_t pinMask) {
        const uint32_t raw = DecodeTelemetryBits(buffer, count, pinMask);
        if (raw == DSHOT_TELEMETRY_NOEDGE || raw == DSHOT_TELEMETRY_INVALID) {
            return -1.0f;
        }
        if (raw == 0x0FFFu) {
            return 0.0f;
        }
        const uint32_t period = (raw & 0x01FFu) << ((raw & 0xFE00u) >> 9u);
        if (period == 0u) {
            return -1.0f;
        }
        const uint32_t erpm100 = (1000000u * 60u / 100u + period / 2u) / period;
        return static_cast<float>(erpm100) * 100.0f /
               static_cast<float>(MOTOR_POLE_PAIR);
    }

    void DecodeReceivedPort(size_t portQueueIndex) {
        if (portQueueIndex >= gpioPortsQueue.size()) return;
        DSHOT_GPIO_PortPins &entry = gpioPortsQueue[portQueueIndex];
        for (auto &pin : entry.pins) {
            const uint8_t id = pin.first;
            float rpm = DecodeRPM(rxBuffer, DSHOT_RX_LENGTH, BSP_DHOSTPinList[id]);
            RPMValid[id] = rpm >= 0.0f;
            if (rpm >= 0.0f) {
                if (pin3DMode[id]) {
                    const uint16_t throttle = pin.second >> 5u;
                    if (throttle > 0u && throttle < DSHOT_3D_FORWARD_MIN_THROTTLE) {
                        rpm = -rpm;
                    }
                }
                RPMs[id] = rpm;
            }
        }
    }

    /*** 接收与解码函数END ***/


public:
    DSHOT_Port(const DSHOT_Port&) = delete;
    DSHOT_Port& operator=(const DSHOT_Port&) = delete;

    static DSHOT_Port& GetInstance() {
        static DSHOT_Port instance;
        return instance;
    }

    void AddPin(GPIO_TypeDef *port, uint16_t ID) {
        for(auto &data : gpioPortsQueue) {
            if(data.port == port) {
                data.pins[ID] = 0;
                return;
            }
        }
        if(!gpioPortsQueue.full()) {
            gpioPortsQueue.push_back(DSHOT_GPIO_PortPins(port, ID));
        }
        else {
            // 如果队列已满，可以选择覆盖最旧的元素或抛出异常
            gpioPortsQueue[0] = DSHOT_GPIO_PortPins(port, ID);
        }
    }

    void SetDataPack(GPIO_TypeDef *port, uint16_t ID, uint16_t DshotDataPack) {
        for(auto &data : gpioPortsQueue) {
            if(data.port == port && data.pins.count(ID)) {
                data.pins[ID] = DshotDataPack;
                return;
            }
        }
    }

    bool InvertState() const{
        return isInverted;
    }

    void TransmitMessage() {
        if (currentIndex >= gpioPortsQueue.size()) {
            return; // 队列已遍历完毕
        }

        if(isInverted) {
            uint16_t pinMusk = 0;
            for(auto &pin : gpioPortsQueue[currentIndex].pins) {
                pinMusk |= BSP_DHOSTPinList[pin.first];
            }
            BSP_DSHOTs::GetInstance().SetPinsOutput(gpioPortsQueue[currentIndex].port, pinMusk);
        }

        GenerateMessage(gpioPortsQueue[currentIndex].pins);
        BSP_DSHOTs::GetInstance().TransmitMessage(outPutData, gpioPortsQueue[currentIndex].port, DSHOT_PACK_LENGTH * 3);
        currentIndex++;
    }

    void ReceiveMessage() {
        uint16_t pinMusk = 0;
        for(auto &pin : gpioPortsQueue[currentIndex-1].pins) {
            pinMusk |= BSP_DHOSTPinList[pin.first];
        }
        BSP_DSHOTs::GetInstance().SetPinsInput(gpioPortsQueue[currentIndex-1].port, pinMusk);
        BSP_DSHOTs::GetInstance().RecceiveMessage((uint32_t*)rxBuffer, gpioPortsQueue[currentIndex-1].port, DSHOT_RX_LENGTH);
    }

    /*** new add ***/
    void Deccode() {
        DecodeReceivedPort(currentIndex-1);
    }

    float getRPM(uint16_t ID) {
        if(ID == 0 || ID > DSHOT_PIN_NUMBER) {
            return -1.0f; // 无效ID
        }
        return RPMs[ID];
    }
    bool isRPMValid(uint16_t ID) const {
        return ID > 0 && ID <= DSHOT_PIN_NUMBER && RPMValid[ID];
    }
    void SetPin3DMode(uint16_t ID, bool enable) {
        if (ID > 0 && ID <= DSHOT_PIN_NUMBER) {
            pin3DMode[ID] = enable;
        }
    }
    /*** end ***/

    void Handle() override {
        currentIndex = 0; // 每次处理时从头开始
        TransmitMessage();
    }
};



template <uint16_t ID>
class DSHOT_PIN {
public:
    DSHOT_PIN(const DSHOT_PIN &) = delete;
    DSHOT_PIN &operator=(const DSHOT_PIN &) = delete;
    static DSHOT_PIN &GetInstance() {
        static DSHOT_PIN instance;
        return instance;
    }

    float GetRPM() {
        return DSHOT_Port::GetInstance().getRPM(ID);
    }

    bool IsRPMValid() const {
        return DSHOT_Port::GetInstance().isRPMValid(ID);
    }

    void SetIntThrottle(int integerThrottle) {
        if(!isMotorEnabled) {
            targetThrottle = 0;
            GenerateDshotPack();
            return;
        }

        targetThrottle = integerThrottle;
        GenerateDshotPack();
    }

    void SetTargetThrottle(float throttle) {
        if(!isMotorEnabled) {
            targetThrottle = 0;
            GenerateDshotPack();
            return;
        }
        // range of throttle 3D(-1，1), normal(0，1)
        if(is3DModeEnabled) {
            if (fabsf(throttle) < 1e-6f) {
                targetThrottle = 0;
            }
            else {
                if(throttle < 0) {
                    targetThrottle = DSHOT_MIN_THROTTLE + static_cast<int>(roundf(fabsf(throttle) * (DSHOT_3D_BACKWARD_MAX_THROTTLE - DSHOT_MIN_THROTTLE)));
                    targetThrottle = targetThrottle < DSHOT_MIN_THROTTLE ? DSHOT_MIN_THROTTLE : targetThrottle;
                    targetThrottle = targetThrottle > DSHOT_3D_BACKWARD_MAX_THROTTLE ? DSHOT_3D_BACKWARD_MAX_THROTTLE : targetThrottle;
                }
                else {
                    targetThrottle = DSHOT_3D_FORWARD_MIN_THROTTLE + static_cast<int>(roundf(throttle * (DSHOT_MAX_THROTTLE - DSHOT_3D_FORWARD_MIN_THROTTLE)));
                    targetThrottle = targetThrottle < DSHOT_3D_FORWARD_MIN_THROTTLE ? DSHOT_3D_FORWARD_MIN_THROTTLE : targetThrottle;
                    targetThrottle = targetThrottle > DSHOT_MAX_THROTTLE ? DSHOT_MAX_THROTTLE : targetThrottle;
                }
            }
        }
        else {
            int integerThrottle = static_cast<int>(roundf(throttle * DSHOT_RANGE));
            targetThrottle = integerThrottle > DSHOT_RANGE ? DSHOT_MAX_THROTTLE : (integerThrottle <= 0 ? 0 : integerThrottle + DSHOT_MIN_THROTTLE); // 1-47保留为特殊指令
        }

        GenerateDshotPack();
    }

    void Set3DMode(bool isEnable) {
        is3DModeEnabled = isEnable;
        DSHOT_Port::GetInstance().SetPin3DMode(ID, isEnable);
    }

    void SetTelemetry(bool isEnable) {
        telemetryRequest = isEnable;
    }

    void EnableMotor() {
        isMotorEnabled = true;
    }

    void DisableMotor() {
        isMotorEnabled = false;
        targetThrottle = 0;
        GenerateDshotPack();
    }

private:
    uint16_t targetThrottle = 0;
    bool is3DModeEnabled = true;
    bool isInverted = true;
    bool telemetryRequest = true;
    bool isMotorEnabled = false; // 是否允许电机工作

    DSHOT_PIN() {
        BSP_DSHOT<ID>::GetInstance();
        DSHOT_Port::GetInstance().AddPin(BSP_DHOSTPortList[ID], ID);
        isInverted = DSHOT_Port::GetInstance().InvertState();//同一个GPIO PORT下不同的PIN必须保持一致
    }

    void GenerateDshotPack() {
        // 构建完整DShot数据包（11位油门 + 1位遥测 + 4位CRC）
        uint16_t packet{0};
        packet |= (targetThrottle << 5);
        packet |= (telemetryRequest ? 1<<4 : 0<<4);

        // 计算CRC并合并到数据包
        uint8_t crc = (packet >> 4) ^ (packet >> 8) ^ (packet >> 12);  // 取前12位计算
        if(isInverted) {
            crc = ~crc;
        }
        packet |= crc & 0x0F;

        DSHOT_Port::GetInstance().SetDataPack(BSP_DHOSTPortList[ID], ID, packet);
    }
};

inline void DMA_XferCpltCallback(DMA_HandleTypeDef *hdma) {
    if (hdma->Instance == DSHOT_TRANSMIT_DMA_HANDLE.Instance) {
        if(DSHOT_Port::GetInstance().InvertState()) {
            DSHOT_Port::GetInstance().ReceiveMessage();
        }
        else {
            DSHOT_Port::GetInstance().TransmitMessage();
        }
    }
    else if (hdma->Instance == DSHOT_RECEIVE_DMA_HANDLE.Instance) {
        if(DSHOT_Port::GetInstance().InvertState()) {
            DSHOT_Port::GetInstance().Deccode();
            DSHOT_Port::GetInstance().TransmitMessage();
        }
    }
}

#endif
