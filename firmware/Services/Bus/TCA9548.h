#ifndef TCA9548_HPP
#define TCA9548_HPP

#include "i2c.h"
#include <cstdint>

class TCA9548 {
public:
    explicit TCA9548(I2C_HandleTypeDef* hi2c, uint8_t addr7 = 0x70)
            : hi2c_(hi2c), addr7_(addr7) {}

    bool IsOnline(uint32_t trials = 3, uint32_t timeout = 100) const {
        return HAL_I2C_IsDeviceReady(hi2c_, addr7_ << 1, trials, timeout) == HAL_OK;
    }

    bool SelectChannel(uint8_t ch, uint32_t timeout = 100) const {
        if (ch > 7) return false;
        uint8_t cmd = static_cast<uint8_t>(1u << ch);
        return HAL_I2C_Master_Transmit(hi2c_, addr7_ << 1, &cmd, 1, timeout) == HAL_OK;
    }

    bool DisableAll(uint32_t timeout = 100) const {
        uint8_t cmd = 0x00;
        return HAL_I2C_Master_Transmit(hi2c_, addr7_ << 1, &cmd, 1, timeout) == HAL_OK;
    }

    uint8_t GetAddress7() const { return addr7_; }

private:
    I2C_HandleTypeDef* hi2c_;
    uint8_t addr7_;
};

#endif