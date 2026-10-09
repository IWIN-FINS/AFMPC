/*******************************************************************************
 * Copyright (c) 2025.
 * IWIN-FINS Lab, Shanghai Jiao Tong University, Shanghai, China.
 * All rights reserved.
 ******************************************************************************/

#ifndef STREAMER_HPP
#define STREAMER_HPP

#include "ProjectConfig.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "Verification/CRC.h"

class Streamer {
public:
    static constexpr uint8_t PROTOCOL_VERSION = 5;
    static constexpr uint8_t COMMAND_HEAD = 0xAA;
    static constexpr uint8_t COMMAND_TYPE_MPC = 0x00;
    static constexpr uint8_t COMMAND_TAIL = 0xBB;
    static constexpr uint8_t TELEMETRY_HEAD = 0x55;
    static constexpr uint8_t TELEMETRY_TYPE = 0x54;
    static constexpr uint8_t TELEMETRY_MESSAGE_STATE_EXECUTION = 0x01;

    static constexpr uint8_t COMMAND_FLAG_ARMED = 0x01;
    static constexpr uint8_t COMMAND_FLAG_MPC_DIRECT = 0x02;
    static constexpr uint8_t COMMAND_FLAG_YAW_DIRECT = 0x04;
    static constexpr uint8_t COMMAND_FLAG_CALIBRATION_MOTOR = 0x08;
    static constexpr uint8_t COMMAND_FLAG_CALIBRATION_CHANNEL = 0x10;
    static constexpr uint8_t COMMAND_FLAG_CALIBRATION_ROLL_ONLY = 0x20;
    static constexpr uint8_t COMMAND_FLAG_CALIBRATION_PITCH_ONLY = 0x40;
    static constexpr uint8_t COMMAND_FLAG_CALIBRATION_YAW_ONLY = 0x80;

    static constexpr uint8_t TELEMETRY_FLAG_ARMED = 0x01;
    static constexpr uint8_t TELEMETRY_FLAG_MPC_DIRECT = 0x02;
    static constexpr uint8_t TELEMETRY_FLAG_FAILSAFE = 0x04;
    static constexpr uint8_t TELEMETRY_FLAG_YAW_DIRECT = 0x08;
    static constexpr uint8_t TELEMETRY_FLAG_EXECUTION_FEEDBACK = 0x10;
    static constexpr uint8_t TELEMETRY_FLAG_RPM_AVAILABLE = 0x20;

    static constexpr uint8_t COMMAND_STATUS_NONE = 0;
    static constexpr uint8_t COMMAND_STATUS_ACCEPTED = 1;
    static constexpr uint8_t COMMAND_STATUS_REJECTED = 2;

    static constexpr uint32_t COMMAND_REJECT_CRC = 1u << 0u;
    static constexpr uint32_t COMMAND_REJECT_VERSION = 1u << 1u;
    static constexpr uint32_t COMMAND_REJECT_FLAGS = 1u << 2u;
    static constexpr uint32_t COMMAND_REJECT_NONFINITE = 1u << 3u;
    static constexpr uint32_t COMMAND_REJECT_STALE_SEQUENCE = 1u << 4u;
    static constexpr uint32_t COMMAND_REJECT_SESSION_REQUIRES_DISARM = 1u << 5u;
    static constexpr uint32_t COMMAND_REJECT_FORMAT = 1u << 6u;
    static constexpr uint32_t COMMAND_REJECT_CALIBRATION = 1u << 7u;

    static constexpr size_t THRUSTER_COUNT = 8;

    struct JoystickData {
        float yawRate;
        float forwardSpeed;
        float leftRightSpeed;
        float downSpeed;
        float upSpeed;
        uint8_t startButton;
        uint8_t camFillLight;
        uint8_t frameFillLight;
        int8_t camYaw;
        int8_t camPitch;
    } __packed;

    JoystickData joystickData{};

    struct MPCControlData {
        float forward;
        float right;
        float down;
        float yaw;
        uint16_t sequence;
        uint32_t session;
        uint32_t senderTimeMs;
        uint16_t crc;
        bool armed;
        bool yawDirect;
        bool calibrationMotor;
        bool calibrationChannel;
        bool calibrationRollOnly;
        bool calibrationPitchOnly;
        bool calibrationYawOnly;
        uint8_t calibrationMotorIndex;
        float calibrationMotorThrottle;
    };

    struct TelemetryData {
        uint16_t sequence;
        uint32_t tickMs;
        uint8_t state;
        bool armed;
        bool mpcDirect;
        bool yawDirect;
        bool failsafe;
        bool executionFeedbackValid;
        bool rpmAvailable;
        uint8_t rpmValidMask;
        uint8_t commandStatus;
        uint32_t rejectFlags;
        uint32_t lastCommandSession;
        uint16_t lastCommandSequence;
        uint16_t lastCommandCrc;
        uint32_t lastCommandSenderTimeMs;
        uint32_t commandCount;
        uint32_t rejectedCommandCount;
        float quatWxyz[4];
        float angularVelocityXyz[3];
        float linearAccelerationXyz[3];
        float yawRad;
        float depthM;
        float pressurePa;
        float receivedChannels[4];
        float appliedMotorThrottle[THRUSTER_COUNT];
        float motorRpm[THRUSTER_COUNT];
    };

    void Decode(uint8_t* data, uint16_t length) {
        if (DecodeLegacy(data, length)) {
            source = ControlSource::LEGACY;
            return;
        }

        AppendToStream(data, length);
        ParseMPCStream();
    }

    bool IsMPCControlActive() const {
        return source == ControlSource::MPC;
    }

    bool ConsumeMPCCommand(MPCControlData& output) {
        if (!mpcCommandPending) {
            return false;
        }
        output = mpcCommand;
        mpcCommandPending = false;
        return true;
    }

    void RecordActuationRejection(uint32_t rejectFlags) {
        if (diagnostics.status != COMMAND_STATUS_ACCEPTED) {
            return;
        }
        diagnostics.status = COMMAND_STATUS_REJECTED;
        diagnostics.rejectFlags = rejectFlags;
        ++rejectedCommandCount;
    }

    void PopulateCommandDiagnostics(TelemetryData& output) const {
        output.commandStatus = diagnostics.status;
        output.rejectFlags = diagnostics.rejectFlags;
        output.lastCommandSession = diagnostics.session;
        output.lastCommandSequence = diagnostics.sequence;
        output.lastCommandCrc = diagnostics.crc;
        output.lastCommandSenderTimeMs = diagnostics.senderTimeMs;
        output.commandCount = commandCount;
        output.rejectedCommandCount = rejectedCommandCount;
        for (size_t i = 0; i < 4; ++i) {
            output.receivedChannels[i] = diagnostics.receivedChannels[i];
        }
    }

    static size_t EncodeTelemetry(const TelemetryData& input,
                                  uint8_t* output,
                                  size_t capacity) {
        if (output == nullptr || capacity < sizeof(TelemetryFrame)) {
            return 0;
        }

        TelemetryFrame frame{};
        frame.head = TELEMETRY_HEAD;
        frame.type = TELEMETRY_TYPE;
        frame.messageType = TELEMETRY_MESSAGE_STATE_EXECUTION;
        frame.version = PROTOCOL_VERSION;
        frame.payloadLength = sizeof(TelemetryPayload);
        frame.sequence = input.sequence;
        frame.tickMs = input.tickMs;

        if (input.armed) frame.payload.flags |= TELEMETRY_FLAG_ARMED;
        if (input.mpcDirect) frame.payload.flags |= TELEMETRY_FLAG_MPC_DIRECT;
        if (input.yawDirect) frame.payload.flags |= TELEMETRY_FLAG_YAW_DIRECT;
        if (input.failsafe) frame.payload.flags |= TELEMETRY_FLAG_FAILSAFE;
        if (input.executionFeedbackValid) {
            frame.payload.flags |= TELEMETRY_FLAG_EXECUTION_FEEDBACK;
        }
        if (input.rpmAvailable) {
            frame.payload.flags |= TELEMETRY_FLAG_RPM_AVAILABLE;
        }

        frame.payload.state = input.state;
        frame.payload.commandStatus = input.commandStatus;
        frame.payload.rpmValidMask = input.rpmValidMask;
        frame.payload.rejectFlags = input.rejectFlags;
        frame.payload.lastCommandSession = input.lastCommandSession;
        frame.payload.lastCommandSequence = input.lastCommandSequence;
        frame.payload.lastCommandCrc = input.lastCommandCrc;
        frame.payload.lastCommandSenderTimeMs = input.lastCommandSenderTimeMs;
        frame.payload.commandCount = input.commandCount;
        frame.payload.rejectedCommandCount = input.rejectedCommandCount;

        for (size_t i = 0; i < 4; ++i) {
            frame.payload.quatWxyz[i] = FiniteOrZero(input.quatWxyz[i]);
            frame.payload.receivedChannels[i] = FiniteOrZero(input.receivedChannels[i]);
        }
        for (size_t i = 0; i < 3; ++i) {
            frame.payload.angularVelocityXyz[i] = FiniteOrZero(input.angularVelocityXyz[i]);
            frame.payload.linearAccelerationXyz[i] = FiniteOrZero(input.linearAccelerationXyz[i]);
        }
        frame.payload.yawRad = FiniteOrZero(input.yawRad);
        frame.payload.depthM = FiniteOrZero(input.depthM);
        frame.payload.pressurePa = FiniteOrZero(input.pressurePa);
        for (size_t i = 0; i < THRUSTER_COUNT; ++i) {
            frame.payload.appliedMotorThrottle[i] = FiniteOrZero(input.appliedMotorThrottle[i]);
            frame.payload.motorRpm[i] = FiniteOrZero(input.motorRpm[i]);
        }

        frame.crc = CRC16Calc(reinterpret_cast<uint8_t*>(&frame),
                              sizeof(TelemetryFrame) - sizeof(frame.crc));
        std::memcpy(output, &frame, sizeof(frame));
        return sizeof(frame);
    }

private:
    enum class ControlSource : uint8_t {
        LEGACY,
        MPC
    } source = ControlSource::LEGACY;

    struct MPCControlFrame {
        uint8_t head;
        uint8_t type;
        uint8_t version;
        uint8_t flags;
        uint16_t sequence;
        uint32_t session;
        uint32_t senderTimeMs;
        float forward;
        float right;
        float down;
        float yaw;
        uint8_t reserved[4];
        uint16_t crc;
        uint8_t tail;
    } __packed;

    struct TelemetryPayload {
        uint8_t flags;
        uint8_t state;
        uint8_t commandStatus;
        uint8_t rpmValidMask;
        uint32_t rejectFlags;
        uint32_t lastCommandSession;
        uint16_t lastCommandSequence;
        uint16_t lastCommandCrc;
        uint32_t lastCommandSenderTimeMs;
        uint32_t commandCount;
        uint32_t rejectedCommandCount;
        float quatWxyz[4];
        float angularVelocityXyz[3];
        float linearAccelerationXyz[3];
        float yawRad;
        float depthM;
        float pressurePa;
        float receivedChannels[4];
        float appliedMotorThrottle[THRUSTER_COUNT];
        float motorRpm[THRUSTER_COUNT];
    } __packed;

    struct TelemetryFrame {
        uint8_t head;
        uint8_t type;
        uint8_t messageType;
        uint8_t version;
        uint16_t payloadLength;
        uint16_t sequence;
        uint32_t tickMs;
        TelemetryPayload payload;
        uint16_t crc;
    } __packed;

    struct CommandDiagnostics {
        uint8_t status = COMMAND_STATUS_NONE;
        uint32_t rejectFlags = 0;
        uint32_t session = 0;
        uint16_t sequence = 0;
        uint16_t crc = 0;
        uint32_t senderTimeMs = 0;
        float receivedChannels[4]{};
    };

    static_assert(sizeof(JoystickData) == 25,
                  "Legacy joystick payload layout changed");
    static_assert(sizeof(MPCControlFrame) == 37,
                  "MPC v5 command wire layout changed");
    static_assert(sizeof(TelemetryPayload) == 160,
                  "MPC v4 telemetry payload layout changed");
    static_assert(sizeof(TelemetryFrame) == 174,
                  "MPC v4 telemetry wire layout changed");

    static float FiniteOrZero(float value) {
        return std::isfinite(value) ? value : 0.0f;
    }

    static bool IsNewerSequence(uint16_t current, uint16_t previous) {
        const uint16_t delta = static_cast<uint16_t>(current - previous);
        return delta != 0u && delta < 0x8000u;
    }

    bool DecodeLegacy(uint8_t* data, uint16_t length) {
        constexpr uint16_t LEGACY_FRAME_SIZE = 28;
        if (length != LEGACY_FRAME_SIZE || data[0] != COMMAND_HEAD) {
            return false;
        }
        const uint16_t calculated = CRC16Calc(data, length - 2);
        const uint16_t received = static_cast<uint16_t>(data[length - 2]) |
                                  (static_cast<uint16_t>(data[length - 1]) << 8u);
        if (calculated != received) {
            return false;
        }
        std::memcpy(&joystickData, data + 1, sizeof(JoystickData));
        return true;
    }

    void AppendToStream(const uint8_t* data, size_t length) {
        for (size_t i = 0; i < length; ++i) {
            if (streamLength == sizeof(streamBuffer)) {
                RemoveStreamPrefix(1);
            }
            streamBuffer[streamLength++] = data[i];
        }
    }

    void RecordAttempt(const MPCControlFrame& frame,
                       uint8_t status,
                       uint32_t rejectFlags) {
        diagnostics.status = status;
        diagnostics.rejectFlags = rejectFlags;
        diagnostics.session = frame.session;
        diagnostics.sequence = frame.sequence;
        diagnostics.crc = frame.crc;
        diagnostics.senderTimeMs = frame.senderTimeMs;
        diagnostics.receivedChannels[0] = FiniteOrZero(frame.forward);
        diagnostics.receivedChannels[1] = FiniteOrZero(frame.right);
        diagnostics.receivedChannels[2] = FiniteOrZero(frame.down);
        diagnostics.receivedChannels[3] = FiniteOrZero(frame.yaw);
    }

    void ParseMPCStream() {
        while (streamLength >= 2) {
            size_t start = 0;
            while (start + 1 < streamLength &&
                   !(streamBuffer[start] == COMMAND_HEAD &&
                     streamBuffer[start + 1] == COMMAND_TYPE_MPC)) {
                ++start;
            }
            if (start + 1 >= streamLength) {
                const bool preserveHead = streamBuffer[streamLength - 1] == COMMAND_HEAD;
                if (preserveHead) {
                    streamBuffer[0] = COMMAND_HEAD;
                    streamLength = 1;
                } else {
                    streamLength = 0;
                }
                return;
            }
            if (start > 0) {
                RemoveStreamPrefix(start);
            }
            if (streamLength < sizeof(MPCControlFrame)) {
                return;
            }

            MPCControlFrame frame{};
            std::memcpy(&frame, streamBuffer, sizeof(frame));
            ++commandCount;

            const uint16_t calculated = CRC16Calc(
                reinterpret_cast<uint8_t*>(&frame),
                offsetof(MPCControlFrame, crc));
            uint32_t rejectFlags = 0;
            const bool requestedCalibrationMotor =
                (frame.flags & COMMAND_FLAG_CALIBRATION_MOTOR) != 0;
            const bool requestedCalibrationChannel =
                (frame.flags & COMMAND_FLAG_CALIBRATION_CHANNEL) != 0;
            const bool requestedCalibrationRollOnly =
                (frame.flags & COMMAND_FLAG_CALIBRATION_ROLL_ONLY) != 0;
            const bool requestedCalibrationPitchOnly =
                (frame.flags & COMMAND_FLAG_CALIBRATION_PITCH_ONLY) != 0;
            const bool requestedCalibrationYawOnly =
                (frame.flags & COMMAND_FLAG_CALIBRATION_YAW_ONLY) != 0;
            constexpr uint8_t knownFlags = COMMAND_FLAG_ARMED |
                                           COMMAND_FLAG_MPC_DIRECT |
                                           COMMAND_FLAG_YAW_DIRECT |
                                           COMMAND_FLAG_CALIBRATION_MOTOR |
                                           COMMAND_FLAG_CALIBRATION_CHANNEL |
                                           COMMAND_FLAG_CALIBRATION_ROLL_ONLY |
                                           COMMAND_FLAG_CALIBRATION_PITCH_ONLY |
                                           COMMAND_FLAG_CALIBRATION_YAW_ONLY;
            if (calculated != frame.crc) {
                rejectFlags |= COMMAND_REJECT_CRC;
            }
            if (frame.version != PROTOCOL_VERSION) {
                rejectFlags |= COMMAND_REJECT_VERSION;
            }
            if (frame.tail != COMMAND_TAIL ||
                frame.reserved[0] != 0 || frame.reserved[1] != 0 ||
                frame.reserved[2] != 0 || frame.reserved[3] != 0) {
                rejectFlags |= COMMAND_REJECT_FORMAT;
            }
            if ((frame.flags & COMMAND_FLAG_MPC_DIRECT) == 0) {
                rejectFlags |= COMMAND_REJECT_FLAGS;
            }
            if ((frame.flags & static_cast<uint8_t>(~knownFlags)) != 0) {
                rejectFlags |= COMMAND_REJECT_FLAGS;
            }
            if (!std::isfinite(frame.forward) ||
                !std::isfinite(frame.right) ||
                !std::isfinite(frame.down) ||
                !std::isfinite(frame.yaw)) {
                rejectFlags |= COMMAND_REJECT_NONFINITE;
            }
            float roundedMotor = 0.0f;
            const uint8_t calibrationModeCount =
                static_cast<uint8_t>(requestedCalibrationMotor) +
                static_cast<uint8_t>(requestedCalibrationChannel) +
                static_cast<uint8_t>(requestedCalibrationRollOnly) +
                static_cast<uint8_t>(requestedCalibrationPitchOnly) +
                static_cast<uint8_t>(requestedCalibrationYawOnly);
            if (calibrationModeCount > 1U) {
                rejectFlags |= COMMAND_REJECT_CALIBRATION;
            }
            if (requestedCalibrationMotor && rejectFlags == 0) {
                roundedMotor = std::round(frame.forward);
                if (roundedMotor < 1.0f || roundedMotor > 8.0f ||
                    std::fabs(frame.forward - roundedMotor) > 1.0e-4f ||
                    std::fabs(frame.right) > 0.100001f ||
                    std::fabs(frame.down) > 1.0e-6f ||
                    std::fabs(frame.yaw) > 1.0e-6f) {
                    rejectFlags |= COMMAND_REJECT_CALIBRATION;
                }
            }
            if (requestedCalibrationChannel && rejectFlags == 0 &&
                (std::fabs(frame.forward) > 0.100001f ||
                 std::fabs(frame.right) > 0.100001f ||
                 std::fabs(frame.down) > 0.100001f ||
                 std::fabs(frame.yaw) > 0.100001f)) {
                rejectFlags |= COMMAND_REJECT_CALIBRATION;
            }
            if ((requestedCalibrationRollOnly || requestedCalibrationPitchOnly ||
                 requestedCalibrationYawOnly) &&
                rejectFlags == 0 &&
                (((requestedCalibrationYawOnly &&
                   (frame.flags & COMMAND_FLAG_YAW_DIRECT) != 0) ||
                  ((requestedCalibrationRollOnly || requestedCalibrationPitchOnly) &&
                   (frame.flags & COMMAND_FLAG_YAW_DIRECT) == 0)) ||
                 std::fabs(frame.forward) > 1.0e-6f ||
                 std::fabs(frame.right) > 1.0e-6f ||
                 std::fabs(frame.down) > 1.0e-6f ||
                 std::fabs(frame.yaw) > 1.0e-6f)) {
                rejectFlags |= COMMAND_REJECT_CALIBRATION;
            }

            const bool requestedArmed = (frame.flags & COMMAND_FLAG_ARMED) != 0;
            if (rejectFlags == 0) {
                if (!hasSession || frame.session != activeSession) {
                    if (requestedArmed) {
                        rejectFlags |= COMMAND_REJECT_SESSION_REQUIRES_DISARM;
                    } else {
                        activeSession = frame.session;
                        hasSession = true;
                        hasAcceptedSequence = false;
                    }
                }
                if (rejectFlags == 0 && hasAcceptedSequence &&
                    !IsNewerSequence(frame.sequence, lastAcceptedSequence)) {
                    rejectFlags |= COMMAND_REJECT_STALE_SEQUENCE;
                }
            }

            if (rejectFlags != 0) {
                ++rejectedCommandCount;
                RecordAttempt(frame, COMMAND_STATUS_REJECTED, rejectFlags);
                RemoveStreamPrefix(
                    (rejectFlags & COMMAND_REJECT_CRC) != 0
                        ? 1
                        : sizeof(frame));
                continue;
            }

            lastAcceptedSequence = frame.sequence;
            hasAcceptedSequence = true;
            mpcCommand.forward = frame.forward;
            mpcCommand.right = frame.right;
            mpcCommand.down = frame.down;
            mpcCommand.yaw = frame.yaw;
            mpcCommand.sequence = frame.sequence;
            mpcCommand.session = frame.session;
            mpcCommand.senderTimeMs = frame.senderTimeMs;
            mpcCommand.crc = frame.crc;
            mpcCommand.armed = requestedArmed;
            mpcCommand.yawDirect = (frame.flags & COMMAND_FLAG_YAW_DIRECT) != 0;
            mpcCommand.calibrationMotor = requestedCalibrationMotor;
            mpcCommand.calibrationChannel = requestedCalibrationChannel;
            mpcCommand.calibrationRollOnly = requestedCalibrationRollOnly;
            mpcCommand.calibrationPitchOnly = requestedCalibrationPitchOnly;
            mpcCommand.calibrationYawOnly = requestedCalibrationYawOnly;
            mpcCommand.calibrationMotorIndex = requestedCalibrationMotor
                ? static_cast<uint8_t>(roundedMotor - 1.0f)
                : 0u;
            mpcCommand.calibrationMotorThrottle = requestedCalibrationMotor
                ? frame.right
                : 0.0f;
            mpcCommandPending = true;
            source = ControlSource::MPC;
            RecordAttempt(frame, COMMAND_STATUS_ACCEPTED, 0);
            RemoveStreamPrefix(sizeof(frame));
        }
    }

    void RemoveStreamPrefix(size_t count) {
        if (count >= streamLength) {
            streamLength = 0;
            return;
        }
        std::memmove(streamBuffer,
                     streamBuffer + count,
                     streamLength - count);
        streamLength -= count;
    }

    MPCControlData mpcCommand{};
    CommandDiagnostics diagnostics{};
    bool mpcCommandPending = false;
    bool hasSession = false;
    bool hasAcceptedSequence = false;
    uint32_t activeSession = 0;
    uint16_t lastAcceptedSequence = 0;
    uint32_t commandCount = 0;
    uint32_t rejectedCommandCount = 0;
    uint8_t streamBuffer[96]{};
    size_t streamLength = 0;
};

#endif // STREAMER_HPP
