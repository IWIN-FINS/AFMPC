#ifndef MS5837_TCA_ARRAY_HPP
#define MS5837_TCA_ARRAY_HPP

#include "PressureSensor/PressureSensorBase.hpp"
#include "Bus/TCA9548.h"
#include "i2c.h"
#include "main.h"
#include <array>
#include <cstdint>
#include <cstring>

class MS5837_TCA_Array {
public:
    static constexpr uint8_t MAX_SENSOR_COUNT = 4;
    static constexpr uint8_t DEFAULT_MS5837_ADDR_1 = 0x76;
    static constexpr uint8_t DEFAULT_MS5837_ADDR_2 = 0x77;

    struct Config {
        I2C_HandleTypeDef* hi2c = &hi2c2;
        uint8_t tca_addr7 = 0x70;

        uint8_t sensor_count = 4;
        std::array<uint8_t, MAX_SENSOR_COUNT> channels = {0, 1, 2, 3};

        uint16_t zero_cal_samples = 64;
        uint8_t filter_window = 8;

        uint8_t sensor_addr_primary = DEFAULT_MS5837_ADDR_1;
        uint8_t sensor_addr_secondary = DEFAULT_MS5837_ADDR_2;

        uint32_t i2c_timeout_ms = 100;
        uint32_t convert_delay_ms = 10;   // OSR4096 可先固定 10ms
    };

    struct SensorData {
        bool online = false;
        bool initialized = false;
        bool data_ready = false;

        uint8_t channel = 0;
        uint8_t addr7 = 0;

        uint16_t prom[7] = {0};

        uint32_t d1_raw = 0;
        uint32_t d2_raw = 0;

        // 这里 offset_pa 代表“空气中绝对压力均值”
        // 也就是 A 板当前 V5_SUB 里 pressureSensorOffset[] 期待的那种量纲
        float offset_pa = 0.0f;

        // 原始计算值
        float absolute_pressure_pa = 0.0f;
        float gauge_pressure_pa = 0.0f;
        float depth_cm = 0.0f;
        float temperature_c = 0.0f;

        // 滤波后
        float absolute_pressure_pa_filtered = 0.0f;
        float gauge_pressure_pa_filtered = 0.0f;
        float depth_cm_filtered = 0.0f;
        float temperature_c_filtered = 0.0f;

        float frequency_hz = 0.0f;
    };

    explicit MS5837_TCA_Array(const Config& cfg)
        : cfg_(cfg), tca_(cfg.hi2c, cfg.tca_addr7) {
        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            sensors_[i].channel = cfg_.channels[i];
            filter_states_[i].Reset();
        }
    }

    // --------------------------------------------------
    // 生命周期接口（V1：在 Setup/Loop 中调用）
    // --------------------------------------------------
    bool Init() {
        online_count_ = 0;
        initialized_ = false;

        if (!tca_.IsOnline()) {
            return false;
        }

        DetectSensors();
        InitSensors();
        ZeroCalibrate();

        initialized_ = (online_count_ > 0);
        last_freq_tick_ = HAL_GetTick();
        return initialized_;
    }

    void Poll() {
        if (!initialized_) return;

        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            if (!sensors_[i].initialized) continue;
            ReadAndUpdateOne(i);
        }

        UpdateFrequency();
    }

    // --------------------------------------------------
    // 对外查询接口（上层控制 / 调试）
    // --------------------------------------------------
    bool IsInitialized() const { return initialized_; }
    bool IsTCAOnline() const { return tca_.IsOnline(); }

    uint8_t GetSensorCount() const { return cfg_.sensor_count; }
    uint8_t GetOnlineCount() const { return online_count_; }

    bool IsOnline(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].online : false;
    }

    bool IsDataReady(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].data_ready : false;
    }

    uint8_t GetChannel(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].channel : 0;
    }

    uint8_t GetAddress7(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].addr7 : 0;
    }

    // 这个接口给 A 板现有深度控制用：返回绝对压力 Pa
    float GetAbsolutePressurePa(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].absolute_pressure_pa_filtered : 0.0f;
    }

    // 这个接口给调试 / 后续新控制逻辑用：返回表压 Pa
    float GetGaugePressurePa(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].gauge_pressure_pa_filtered : 0.0f;
    }

    float GetDepthCm(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].depth_cm_filtered : 0.0f;
    }

    float GetTemperatureC(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].temperature_c_filtered : 0.0f;
    }

    float GetOffsetPa(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].offset_pa : 0.0f;
    }

    float GetFrequencyHz(uint8_t idx) const {
        return IsIndexValid(idx) ? sensors_[idx].frequency_hz : 0.0f;
    }

    const SensorData& GetSensorData(uint8_t idx) const {
        return sensors_[idx];
    }

private:
    struct FilterState {
        static constexpr uint8_t MAX_WIN = 16;

        std::array<float, MAX_WIN> p_abs_hist{};
        std::array<float, MAX_WIN> p_gauge_hist{};
        std::array<float, MAX_WIN> depth_hist{};
        std::array<float, MAX_WIN> temp_hist{};

        float p_abs_sum = 0.0f;
        float p_gauge_sum = 0.0f;
        float depth_sum = 0.0f;
        float temp_sum = 0.0f;

        uint8_t idx = 0;
        uint8_t cnt = 0;

        void Reset() {
            p_abs_hist.fill(0.0f);
            p_gauge_hist.fill(0.0f);
            depth_hist.fill(0.0f);
            temp_hist.fill(0.0f);
            p_abs_sum = 0.0f;
            p_gauge_sum = 0.0f;
            depth_sum = 0.0f;
            temp_sum = 0.0f;
            idx = 0;
            cnt = 0;
        }

        void Push(uint8_t win, float p_abs, float p_gauge, float depth, float temp,
                  float& p_abs_out, float& p_gauge_out, float& depth_out, float& temp_out) {
            if (win == 0) win = 1;
            if (win > MAX_WIN) win = MAX_WIN;

            if (cnt < win) {
                p_abs_hist[idx] = p_abs;
                p_gauge_hist[idx] = p_gauge;
                depth_hist[idx] = depth;
                temp_hist[idx] = temp;

                p_abs_sum += p_abs;
                p_gauge_sum += p_gauge;
                depth_sum += depth;
                temp_sum += temp;

                cnt++;
                idx = static_cast<uint8_t>((idx + 1) % win);

                p_abs_out = p_abs_sum / cnt;
                p_gauge_out = p_gauge_sum / cnt;
                depth_out = depth_sum / cnt;
                temp_out = temp_sum / cnt;
            } else {
                p_abs_sum -= p_abs_hist[idx];
                p_gauge_sum -= p_gauge_hist[idx];
                depth_sum -= depth_hist[idx];
                temp_sum -= temp_hist[idx];

                p_abs_hist[idx] = p_abs;
                p_gauge_hist[idx] = p_gauge;
                depth_hist[idx] = depth;
                temp_hist[idx] = temp;

                p_abs_sum += p_abs;
                p_gauge_sum += p_gauge;
                depth_sum += depth;
                temp_sum += temp;

                idx = static_cast<uint8_t>((idx + 1) % win);

                p_abs_out = p_abs_sum / win;
                p_gauge_out = p_gauge_sum / win;
                depth_out = depth_sum / win;
                temp_out = temp_sum / win;
            }
        }
    };

private:
    bool IsIndexValid(uint8_t idx) const {
        return idx < cfg_.sensor_count;
    }

    void DetectSensors() {
        online_count_ = 0;

        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            auto& s = sensors_[i];
            s.online = false;
            s.initialized = false;
            s.data_ready = false;

            if (!tca_.SelectChannel(s.channel, cfg_.i2c_timeout_ms)) {
                continue;
            }
            HAL_Delay(2);

            if (IsDeviceReady(cfg_.sensor_addr_primary)) {
                s.addr7 = cfg_.sensor_addr_primary;
            } else if (IsDeviceReady(cfg_.sensor_addr_secondary)) {
                s.addr7 = cfg_.sensor_addr_secondary;
            } else {
                continue;
            }

            s.online = true;
            online_count_++;
        }
    }

    void InitSensors() {
        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            auto& s = sensors_[i];
            if (!s.online) continue;

            if (!SelectSensorChannel(i)) continue;
            if (!ResetSensor(s)) continue;
            if (!ReadPROM(s)) continue;

            s.initialized = true;
        }
    }

    void ZeroCalibrate() {
        // V1：校准的是“空气绝对压力均值”，用于适配 A 板现有 offset 思路
        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            sensors_[i].offset_pa = 0.0f;
        }

        for (uint16_t k = 0; k < cfg_.zero_cal_samples; ++k) {
            for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
                auto& s = sensors_[i];
                if (!s.initialized) continue;

                float abs_pa = 0.0f;
                float temp_c = 0.0f;
                if (ReadAbsolutePressureAndTemp(s, abs_pa, temp_c)) {
                    s.offset_pa += abs_pa;
                }
            }
        }

        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            auto& s = sensors_[i];
            if (!s.initialized) continue;
            s.offset_pa /= static_cast<float>(cfg_.zero_cal_samples);
        }
    }

    void ReadAndUpdateOne(uint8_t idx) {
        auto& s = sensors_[idx];

        float abs_pa = 0.0f;
        float temp_c = 0.0f;
        if (!ReadAbsolutePressureAndTemp(s, abs_pa, temp_c)) {
            return;
        }

        s.absolute_pressure_pa = abs_pa;
        s.gauge_pressure_pa = abs_pa - s.offset_pa;
        s.depth_cm = s.gauge_pressure_pa / 98.0f;
        s.temperature_c = temp_c;

        filter_states_[idx].Push(
            cfg_.filter_window,
            s.absolute_pressure_pa,
            s.gauge_pressure_pa,
            s.depth_cm,
            s.temperature_c,
            s.absolute_pressure_pa_filtered,
            s.gauge_pressure_pa_filtered,
            s.depth_cm_filtered,
            s.temperature_c_filtered
        );

        s.data_ready = true;
        sample_counter_[idx]++;
    }

    bool SelectSensorChannel(uint8_t idx) {
        return tca_.SelectChannel(sensors_[idx].channel, cfg_.i2c_timeout_ms);
    }

    bool IsDeviceReady(uint8_t addr7) const {
        return HAL_I2C_IsDeviceReady(cfg_.hi2c, addr7 << 1, 3, cfg_.i2c_timeout_ms) == HAL_OK;
    }

    bool WriteCmd(uint8_t addr7, uint8_t cmd) const {
        return HAL_I2C_Master_Transmit(cfg_.hi2c, addr7 << 1, &cmd, 1, cfg_.i2c_timeout_ms) == HAL_OK;
    }

    bool ReadBytes(uint8_t addr7, uint8_t* buf, uint16_t len) const {
        return HAL_I2C_Master_Receive(cfg_.hi2c, addr7 << 1, buf, len, cfg_.i2c_timeout_ms) == HAL_OK;
    }

    bool ResetSensor(const SensorData& s) {
        uint8_t cmd = 0x1E;
        if (HAL_I2C_Master_Transmit(cfg_.hi2c, s.addr7 << 1, &cmd, 1, cfg_.i2c_timeout_ms) != HAL_OK) {
            return false;
        }
        HAL_Delay(20);
        return true;
    }

    bool ReadPROM(SensorData& s) {
        for (int i = 0; i < 7; ++i) {
            uint8_t cmd = static_cast<uint8_t>(0xA0 + i * 2);
            uint8_t rx[2] = {0};

            if (!WriteCmd(s.addr7, cmd)) return false;
            if (!ReadBytes(s.addr7, rx, 2)) return false;

            s.prom[i] = static_cast<uint16_t>((rx[0] << 8) | rx[1]);
        }
        return true;
    }

    bool ReadADC(const SensorData& s, uint8_t conv_cmd, uint32_t& adc_out) {
        if (!WriteCmd(s.addr7, conv_cmd)) {
            return false;
        }

        HAL_Delay(cfg_.convert_delay_ms);

        uint8_t adc_cmd = 0x00;
        uint8_t rx[3] = {0};

        if (HAL_I2C_Master_Transmit(cfg_.hi2c, s.addr7 << 1, &adc_cmd, 1, cfg_.i2c_timeout_ms) != HAL_OK) {
            return false;
        }
        if (HAL_I2C_Master_Receive(cfg_.hi2c, s.addr7 << 1, rx, 3, cfg_.i2c_timeout_ms) != HAL_OK) {
            return false;
        }

        adc_out = (static_cast<uint32_t>(rx[0]) << 16) |
            (static_cast<uint32_t>(rx[1]) << 8)  |
            static_cast<uint32_t>(rx[2]);
        return true;
    }

    bool ReadAbsolutePressureAndTemp(SensorData& s, float& abs_pa_out, float& temp_c_out) {
        if (!tca_.SelectChannel(s.channel, cfg_.i2c_timeout_ms)) {
            return false;
        }

        uint32_t adc_temp = 0;
        uint32_t adc_press = 0;

        if (!ReadADC(s, 0x58, adc_temp)) return false;   // D2
        if (!ReadADC(s, 0x48, adc_press)) return false;  // D1

        s.d2_raw = adc_temp;
        s.d1_raw = adc_press;

        // ===== 这里保留 C 板那套 MS5837 补偿公式 =====
        int32_t dT = static_cast<int32_t>(adc_temp) - (static_cast<int32_t>(s.prom[5]) * 256L);

        int64_t SENS = static_cast<int64_t>(s.prom[1]) * 65536LL +
            (static_cast<int64_t>(s.prom[3]) * dT) / 128LL;
        int64_t OFF  = static_cast<int64_t>(s.prom[2]) * 131072LL +
            (static_cast<int64_t>(s.prom[4]) * dT) / 64LL;

        int32_t TEMP = 2000L + (static_cast<int64_t>(dT) * s.prom[6]) / 8388608LL;

        int64_t OFF2  = OFF;
        int64_t SENS2 = SENS;

        if (TEMP < 2000) {
            int64_t OFFi  = (31LL * (TEMP - 2000) * (TEMP - 2000)) / 8LL;
            int64_t SENSi = (63LL * (TEMP - 2000) * (TEMP - 2000)) / 32LL;
            OFF2  -= OFFi;
            SENS2 -= SENSi;
        }

        // MS5837-02BA datasheet: the compensated result has 0.01 mbar
        // resolution.  Since 0.01 mbar == 1 Pa, its numeric value is already
        // pressure in pascals and must not be multiplied by 100 again.
        float pressure_centi_mbar =
            (((static_cast<float>(adc_press) * static_cast<float>(SENS2)) / 2097152.0f) -
             static_cast<float>(OFF2)) / 32768.0f;

        abs_pa_out = pressure_centi_mbar;
        temp_c_out = TEMP / 100.0f;
        return true;
    }

    void UpdateFrequency() {
        uint32_t now = HAL_GetTick();
        if (now - last_freq_tick_ < 1000) return;

        for (uint8_t i = 0; i < cfg_.sensor_count; ++i) {
            sensors_[i].frequency_hz = static_cast<float>(sample_counter_[i]);
            sample_counter_[i] = 0;
        }
        last_freq_tick_ = now;
    }

private:
    Config cfg_;
    TCA9548 tca_;

    std::array<SensorData, MAX_SENSOR_COUNT> sensors_{};
    std::array<FilterState, MAX_SENSOR_COUNT> filter_states_{};
    std::array<uint16_t, MAX_SENSOR_COUNT> sample_counter_{};

    uint8_t online_count_ = 0;
    bool initialized_ = false;
    uint32_t last_freq_tick_ = 0;
};

#endif
