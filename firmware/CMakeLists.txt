cmake_minimum_required(VERSION 3.21)

# Preserve the original Windows ARM Compiler workflow when no toolchain file
# is supplied. Linux builds use cmake/arm-none-eabi-gcc.cmake via a preset.
if(WIN32 AND NOT CMAKE_TOOLCHAIN_FILE)
        set(CMAKE_SYSTEM_NAME Generic)
        set(CMAKE_SYSTEM_VERSION 1)
        set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
        set(CMAKE_C_COMPILER armclang.exe)
        set(CMAKE_CXX_COMPILER armclang.exe)
        set(CMAKE_ASM_COMPILER armclang.exe)
endif()

#*******************************************************************************************#
# 工程设置与添加FineMote文件
project(FineMote C CXX ASM)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        execute_process(
                COMMAND ${CMAKE_C_COMPILER} -dumpmachine
                OUTPUT_VARIABLE FINEMOTE_GNU_TARGET
                OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT FINEMOTE_GNU_TARGET MATCHES "^arm-none-eabi$")
                message(FATAL_ERROR
                        "GNU compiler target '${FINEMOTE_GNU_TARGET}' is not STM32 ARM. "
                        "Use: cmake --preset linux-gcc-robomaster-a")
        endif()
        set(FINEMOTE_GNU_ARM TRUE)
elseif(CMAKE_C_COMPILER_ID MATCHES "ARMClang|Clang")
        set(FINEMOTE_ARMCLANG TRUE)
else()
        message(FATAL_ERROR
                "Unsupported compiler '${CMAKE_C_COMPILER_ID}'. "
                "On Linux use: cmake --preset linux-gcc-robomaster-a")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib) # 对于静态库

function(INCLUDE_SUB_DIR_INC root_dir)
        if(IS_DIRECTORY ${root_dir})
                #找含有.h文件或含有.hpp文件，但该目录不一定是Inc或include目录,则将当前目录添加到头文件搜索路径
                file(GLOB_RECURSE head_files ${root_dir}/*.h ${root_dir}/*.h++ ${root_dir}/*.hpp)
                if(head_files)
                include_directories(${root_dir})
                message(STATUS "Add head file search path: ${root_dir}")
                endif()
                #递归寻找子目录
                file(GLOB sub_directories RELATIVE ${root_dir} ${root_dir}/*)
                foreach(sub_dir ${sub_directories})
                        INCLUDE_SUB_DIR_INC(${root_dir}/${sub_dir})
                endforeach()
        endif()
endfunction()

INCLUDE_SUB_DIR_INC(${PROJECT_SOURCE_DIR}/Interface)
INCLUDE_SUB_DIR_INC(${PROJECT_SOURCE_DIR}/Devices)
INCLUDE_SUB_DIR_INC(${PROJECT_SOURCE_DIR}/Algorithms)
INCLUDE_SUB_DIR_INC(${PROJECT_SOURCE_DIR}/Services)
INCLUDE_SUB_DIR_INC(${PROJECT_SOURCE_DIR}/Components)
INCLUDE_SUB_DIR_INC(${PROJECT_SOURCE_DIR}/Tests)

file(GLOB_RECURSE FINEMOTE_SOURCE_FILES
        "Interface/*.*"
        "Devices/*.*"
        "Algorithms/*.*"
        "Services/*.*"
        "Components/*.*"
        "Tests/*.*"
        )

#*******************************************************************************************#
# 添加CMSIS-DSP库

option(DISABLEFLOAT16 "Disable float16 support" OFF)

include_directories(ThirdParty/CMSIS-DSP/Include)
include_directories(ThirdParty/CMSIS-DSP/Include/dsp)
include_directories(ThirdParty/CMSIS-DSP/PrivateInclude)

file(GLOB_RECURSE CMSIS_DSP_SOURCE_FILES
        "ThirdParty/CMSIS-DSP/Source/BasicMathFunctions/BasicMathFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/BayesFunctions/BayesFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/CommonTables/CommonTables.c"
        "ThirdParty/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/ControllerFunctions/ControllerFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/DistanceFunctions/DistanceFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/FastMathFunctions/FastMathFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/FilteringFunctions/FilteringFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/InterpolationFunctions/InterpolationFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/MatrixFunctions/MatrixFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/QuaternionMathFunctions/QuaternionMathFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/StatisticsFunctions/StatisticsFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/SupportFunctions/SupportFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/SVMFunctions/SVMFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/TransformFunctions/TransformFunctions.c"
        "ThirdParty/CMSIS-DSP/Source/WindowFunctions/WindowFunctions.c"
)

if ((NOT ARMAC5) AND (NOT DISABLEFLOAT16))
file(GLOB_RECURSE CMSIS_DSP_SOURCE_FILES ${CMSIS_DSP_SOURCE_FILES}
        "ThirdParty/CMSIS-DSP/Source/BasicMathFunctions/BasicMathFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/BayesFunctions/BayesFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/CommonTables/CommonTablesF16.c"
        "ThirdParty/CMSIS-DSP/Source/ComplexMathFunctions/ComplexMathFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/DistanceFunctions/DistanceFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/FastMathFunctions/FastMathFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/FilteringFunctions/FilteringFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/InterpolationFunctions/InterpolationFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/MatrixFunctions/MatrixFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/StatisticsFunctions/StatisticsFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/SupportFunctions/SupportFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/SVMFunctions/SVMFunctionsF16.c"
        "ThirdParty/CMSIS-DSP/Source/TransformFunctions/TransformFunctionsF16.c"
)
endif()

# 添加ETL(Embedded Template Library)库

include_directories(ThirdParty/etl/include)

# 添加所有ThirdParty库的源文件

file(GLOB_RECURSE FINEMOTE_COMMON_SOURCE_FILES ${FINEMOTE_SOURCE_FILES} ${CMSIS_DSP_SOURCE_FILES})

#*******************************************************************************************#
# 添加BSP。默认保留旧行为；Linux preset 只构建实艇 Robomaster_A。
set(FINEMOTE_BOARD "ALL" CACHE STRING "BSP target: ALL, MC_Board, Robomaster_C, or Robomaster_A")
set_property(CACHE FINEMOTE_BOARD PROPERTY STRINGS ALL MC_Board Robomaster_C Robomaster_A)

if(FINEMOTE_BOARD STREQUAL "ALL")
        add_subdirectory(BSP/MC_Board)
        add_subdirectory(BSP/Robomaster_C)
        add_subdirectory(BSP/Robomaster_A)
elseif(FINEMOTE_BOARD MATCHES "^(MC_Board|Robomaster_C|Robomaster_A)$")
        add_subdirectory(BSP/${FINEMOTE_BOARD})
else()
        message(FATAL_ERROR "Unknown FINEMOTE_BOARD='${FINEMOTE_BOARD}'")
endif()
