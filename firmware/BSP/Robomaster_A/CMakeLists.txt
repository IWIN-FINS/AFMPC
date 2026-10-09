#*******************************************************************************************#
# 该部分内容需要根据实际情况修改 Begin

# 更换BSP需要修改的部分
set(BOARD_NAME Robomaster_A)
set(CHIP_NAME STM32F427xx)
set(CMAKE_SYSTEM_PROCESSOR cortex-m4)

# 该部分内容需要根据实际情况修改 End
#*******************************************************************************************#
# 添加BSP文件

file(GLOB BSP_INCLUDE_DIRS
        "FineMote_BSP"
        "Drivers/CMSIS/Include"
        "Drivers/STM32F*xx_HAL_Driver/Inc"
        "Drivers/STM32F*xx_HAL_Driver/Inc/Legacy"
        "Drivers/CMSIS/Device/ST/STM32F*xx/Include"
        "Core/Inc"
)

include_directories(. ${BSP_INCLUDE_DIRS})

file(GLOB_RECURSE SOURCES
        ${FINEMOTE_COMMON_SOURCE_FILES}
        "Drivers/*.*"
        "Core/*.*"
        "./*.cpp"
)

file(GLOB_RECURSE EXCLUDE_SRCS
        "STM32*_FLASH.ld"
        "STM32*_RAM.ld"
)

if(FINEMOTE_ARMCLANG)
        file(GLOB ARMCLANG_STARTUP "MDK-ARM/startup_*.s")
        list(APPEND SOURCES ${ARMCLANG_STARTUP})
        file(GLOB_RECURSE ARMCLANG_EXCLUDE_SRCS
                "Core/Startup/*.*"
                "Core/Src/syscalls.c"
                "Core/Src/sysmem.c"
        )
        list(APPEND EXCLUDE_SRCS ${ARMCLANG_EXCLUDE_SRCS})
endif()

# 从源文件列表(SOURCES)中剔除干扰文件(EXCLUDE_SRCS)
list(REMOVE_ITEM SOURCES ${EXCLUDE_SRCS})

#*******************************************************************************************#

# BSP宏定义
string(TOUPPER ${BOARD_NAME} UPPER_BOARD_NAME)

#*******************************************************************************************#
# 生成可执行文件
add_executable(${BOARD_NAME} ${SOURCES} ${LINKER_SCRIPT}
        ../../Services/Bus/TCA9548.h
        ../../Devices/ServoMotors/ServoS30.hpp)

target_compile_definitions(${BOARD_NAME} PRIVATE
        USE_HAL_DRIVER
        ${CHIP_NAME}
        __${UPPER_BOARD_NAME}
        __CMAKE_APP
        ARM_MATH_CM4
        ARM_MATH_MATRIX_CHECK
        ARM_MATH_ROUNDING
)

set(HEX_FILE ${CMAKE_BINARY_DIR}/bin/${BOARD_NAME}.hex)
set(BIN_FILE ${CMAKE_BINARY_DIR}/bin/${BOARD_NAME}.bin)
set_target_properties(${BOARD_NAME} PROPERTIES SUFFIX ".elf")

if(FINEMOTE_GNU_ARM)
        set(CPU_FLAGS -mcpu=${CMAKE_SYSTEM_PROCESSOR} -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard)
        target_compile_options(${BOARD_NAME} PRIVATE
                ${CPU_FLAGS}
                -Os
                -g3
                -ffunction-sections
                -fdata-sections
                -funsigned-char
                -fshort-enums
                $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti;-fno-threadsafe-statics;-fno-use-cxa-atexit>
                $<$<COMPILE_LANGUAGE:ASM>:-x;assembler-with-cpp>
        )

        set(LINKER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/STM32F427IIHX_FLASH.ld)
        target_link_options(${BOARD_NAME} PRIVATE
                ${CPU_FLAGS}
                -T${LINKER_SCRIPT}
                --specs=nano.specs
                --specs=nosys.specs
                -Wl,--gc-sections
                -Wl,-Map=${CMAKE_BINARY_DIR}/bin/${BOARD_NAME}.map
                -Wl,--print-memory-usage
        )
        set_property(TARGET ${BOARD_NAME} APPEND PROPERTY LINK_DEPENDS ${LINKER_SCRIPT})

        add_custom_command(TARGET ${BOARD_NAME} POST_BUILD
                COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:${BOARD_NAME}> ${HEX_FILE}
                COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:${BOARD_NAME}> ${BIN_FILE}
                COMMAND ${CMAKE_SIZE} $<TARGET_FILE:${BOARD_NAME}>
                BYPRODUCTS ${HEX_FILE} ${BIN_FILE}
                COMMENT "Generating ${BOARD_NAME}.hex and ${BOARD_NAME}.bin"
        )
else()
        set(C_TARGET_FLAG --target=arm-arm-none-eabi)
        set(COMPILE_RULE_FLAG "-mcpu=${CMAKE_SYSTEM_PROCESSOR}")
        set(ASM_TARGET_FLAG --target=arm-arm-none-eabi)
        set(LINKER_TARGET_FLAG "--cpu=${CMAKE_SYSTEM_PROCESSOR}")
        set(CMAKE_C_FLAGS "${C_TARGET_FLAG} ${COMPILE_RULE_FLAG} -fno-rtti -funsigned-char -fshort-enums -fshort-wchar -gdwarf-4 -Os -ffunction-sections")
        set(CMAKE_CXX_FLAGS ${CMAKE_C_FLAGS})
        set(CMAKE_ASM_FLAGS "${ASM_TARGET_FLAG} ${COMPILE_RULE_FLAG} -masm=auto -c -gdwarf-4")
        set(SECTIONS_SCRIPT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/MDK-ARM/${BOARD_NAME}/${BOARD_NAME}.sct)
        set(CMAKE_EXE_LINKER_FLAGS "${LINKER_TARGET_FLAG} --strict --scatter ${SECTIONS_SCRIPT_PATH} --info sizes --info totals --info unused --info veneers --summary_stderr --info summarysizes --map --load_addr_map_info --xref --callgraph --symbols --list ${BOARD_NAME}.map")

        add_custom_command(TARGET ${BOARD_NAME} POST_BUILD
                COMMAND fromelf.exe --i32combined --output=${HEX_FILE} $<TARGET_FILE:${BOARD_NAME}>
                COMMENT "Generating ${BOARD_NAME}.hex"
        )
endif()
