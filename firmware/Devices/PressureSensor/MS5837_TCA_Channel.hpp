#ifndef MS5837_TCA_CHANNEL_HPP
#define MS5837_TCA_CHANNEL_HPP

#include "PressureSensor/PressureSensorBase.hpp"
#include "PressureSensor/MS5837_TCA_Array.hpp"
#include <cstdint>

class MS5837_TCA_Channel : public PressureSensorBase {
public:
    MS5837_TCA_Channel(MS5837_TCA_Array& owner, uint8_t index)
        : owner_(owner), index_(index) {
        this->SetDivisionFactor(1);
    }

    void Handle() override {
        // 不做任何阻塞式 I2C，只同步管理器里的缓存数据
        if (!owner_.IsDataReady(index_)) {
            return;
        }

        // 适配 A 板现有深度控制：
        // PressureSensorBase::pressure 这里写入“绝对压力 Pa”
        pressure = owner_.GetAbsolutePressurePa(index_);
        frequency = owner_.GetFrequencyHz(index_);
    }

    bool IsOnline() const {
        return owner_.IsOnline(index_);
    }

    bool IsDataReady() const {
        return owner_.IsDataReady(index_);
    }

    float GetGaugePressurePa() const {
        return owner_.GetGaugePressurePa(index_);
    }

    float GetDepthCm() const {
        return owner_.GetDepthCm(index_);
    }

    float GetTemperatureC() const {
        return owner_.GetTemperatureC(index_);
    }

    float GetOffsetPa() const {
        return owner_.GetOffsetPa(index_);
    }

    uint8_t GetChannel() const {
        return owner_.GetChannel(index_);
    }

private:
    MS5837_TCA_Array& owner_;
    uint8_t index_;
};

#endif