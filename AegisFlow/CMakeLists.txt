################################################################################
# AegisFlow — CMakeLists.txt
#
# Build system for the AegisFlow CICFlowMeter-Compatible Feature Engine.
#
# Standard build:
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#
# With tests:
#   cmake -S . -B build -DBUILD_TESTS=ON
#   cmake --build build -j$(nproc)
#   cd build && ctest -V
#
# With benchmarks:
#   cmake -S . -B build -DBUILD_BENCHMARKS=ON
#   cmake --build build -j$(nproc)
#
# With Address Sanitizer (debug):
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DAESGISFLOW_ASAN=ON
#
################################################################################
cmake_minimum_required(VERSION 3.16)

project(AegisFlow
    VERSION     1.0.0
    DESCRIPTION "High-Performance CICFlowMeter-Compatible Flow Feature Engine"
    LANGUAGES   C
)

################################################################################
# Standards & Global Options
################################################################################
set(CMAKE_C_STANDARD          17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS        OFF)   # no GNU extensions — pure C17

# Coloured compiler diagnostics
if(CMAKE_GENERATOR STREQUAL "Ninja")
    add_compile_options(-fdiagnostics-color=always)
endif()

################################################################################
# Build type default
################################################################################
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()
message(STATUS "[AegisFlow] Build type: ${CMAKE_BUILD_TYPE}")

################################################################################
# Options
################################################################################
option(BUILD_TESTS      "Build AegisFlow unit tests"           OFF)
option(BUILD_BENCHMARKS "Build AegisFlow benchmark utilities"  OFF)
option(BUILD_EXAMPLES   "Build AegisFlow example executables"  ON)
option(AEGISFLOW_ASAN   "Enable AddressSanitizer (Debug only)" OFF)
option(AEGISFLOW_KAFKA  "Build Kafka exporter + aegisflow_kafka binary" OFF)

################################################################################
# Compiler warnings (production quality)
################################################################################
set(AEGISFLOW_WARNINGS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wformat=2
    -Wconversion
    -Wsign-conversion
    -Wmissing-prototypes
    -Wstrict-prototypes
    -Wold-style-definition
    -Wnull-dereference
)

################################################################################
# Sanitizers
################################################################################
if(AEGISFLOW_ASAN)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "AEGISFLOW_ASAN is recommended only with Debug builds")
    endif()
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
    message(STATUS "[AegisFlow] AddressSanitizer + UBSan enabled")
endif()

################################################################################
# Dependencies
################################################################################

# ── FetchContent for header-only and small library deps ──────────────────────
include(FetchContent)

# ── uthash (header-only, BSD-2-Clause) ───────────────────────────────────────
FetchContent_Declare(
    uthash
    GIT_REPOSITORY https://github.com/troydhanson/uthash.git
    GIT_TAG        v2.3.0
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(uthash)
# uthash only ships the header in src/
set(UTHASH_INCLUDE_DIR "${uthash_SOURCE_DIR}/src")
message(STATUS "[AegisFlow] uthash include: ${UTHASH_INCLUDE_DIR}")

# ── cJSON ────────────────────────────────────────────────────────────────────
FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG        v1.7.18
    GIT_SHALLOW    TRUE
)
# Build only the static library (no tests, no shared lib)
set(ENABLE_CJSON_TEST    OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(ENABLE_CJSON_UTILS   OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cjson)
message(STATUS "[AegisFlow] cJSON fetched")

# ── libpcap ──────────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(Libpcap REQUIRED)
message(STATUS "[AegisFlow] libpcap: ${LIBPCAP_LIBRARIES} (${LIBPCAP_INCLUDE_DIR})")

# ── pthreads ─────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Math library ─────────────────────────────────────────────────────────────
find_library(MATH_LIB m REQUIRED)

# ── librdkafka (optional — only when AEGISFLOW_KAFKA=ON) ─────────────────────
if(AEGISFLOW_KAFKA)
    find_path(RDKAFKA_INCLUDE_DIR
        NAMES librdkafka/rdkafka.h
        HINTS /usr/include /usr/local/include /opt/homebrew/include
    )
    find_library(RDKAFKA_LIBRARY
        NAMES rdkafka
        HINTS /usr/lib /usr/local/lib /opt/homebrew/lib
    )
    if(NOT RDKAFKA_INCLUDE_DIR OR NOT RDKAFKA_LIBRARY)
        message(FATAL_ERROR
            "[AegisFlow] librdkafka not found.\n"
            "  macOS:  brew install librdkafka\n"
            "  Ubuntu: apt install librdkafka-dev\n"
            "  Or disable with -DAEGISFLOW_KAFKA=OFF"
        )
    endif()
    message(STATUS "[AegisFlow] librdkafka: ${RDKAFKA_LIBRARY} (${RDKAFKA_INCLUDE_DIR})")
endif()

################################################################################
# AegisFlow core STATIC library
#
# Includes all engine components except main.c.
# Tests, examples, and benchmarks link against this.
################################################################################
add_library(aegisflow_core STATIC
    src/utils/utils.c
    src/flow/flow.c
    src/flow/flow_table.c
    src/features/features.c
    src/exporter/json_exporter.c
    src/exporter/csv_exporter.c
    src/capture/capture.c
    src/aegisflow.c
)

target_include_directories(aegisflow_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${UTHASH_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${LIBPCAP_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${cjson_SOURCE_DIR}>
)

target_link_libraries(aegisflow_core
    PUBLIC
        cjson
        ${LIBPCAP_LIBRARIES}
        Threads::Threads
        ${MATH_LIB}
)

target_compile_options(aegisflow_core PRIVATE ${AEGISFLOW_WARNINGS})

target_compile_definitions(aegisflow_core
    PRIVATE
        $<$<CONFIG:Debug>:AEGISFLOW_DEBUG>
        _POSIX_C_SOURCE=200809L
        _DEFAULT_SOURCE
)

################################################################################
# Main CLI executable
################################################################################
add_executable(aegisflow src/main.c)
target_link_libraries(aegisflow PRIVATE aegisflow_core)
target_compile_options(aegisflow PRIVATE ${AEGISFLOW_WARNINGS})

install(TARGETS aegisflow RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include/aegisflow)

################################################################################
# Kafka exporter target (optional — requires AEGISFLOW_KAFKA=ON)
#
# Builds:
#   - aegisflow_kafka_exporter  static library (kafka_exporter.c)
#   - aegisflow_kafka           CLI binary     (kafka_main.c)
#
# Build:
#   cmake -S . -B build -DAEGISFLOW_KAFKA=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --target aegisflow_kafka
################################################################################
if(AEGISFLOW_KAFKA)
    # Static library: kafka_exporter.c (links against rdkafka + core)
    add_library(aegisflow_kafka_exporter STATIC
        src/exporter/kafka_exporter.c
    )

    target_include_directories(aegisflow_kafka_exporter
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${RDKAFKA_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${cjson_SOURCE_DIR}>
    )

    target_link_libraries(aegisflow_kafka_exporter
        PUBLIC
            aegisflow_core
            ${RDKAFKA_LIBRARY}
    )

    target_compile_options(aegisflow_kafka_exporter PRIVATE ${AEGISFLOW_WARNINGS})
    target_compile_definitions(aegisflow_kafka_exporter
        PRIVATE
            _POSIX_C_SOURCE=200809L
            _DEFAULT_SOURCE
    )

    # Executable: kafka_main.c
    add_executable(aegisflow_kafka src/kafka_main.c)
    target_link_libraries(aegisflow_kafka PRIVATE aegisflow_kafka_exporter)
    target_compile_options(aegisflow_kafka PRIVATE ${AEGISFLOW_WARNINGS})
    target_compile_definitions(aegisflow_kafka
        PRIVATE
            _POSIX_C_SOURCE=200809L
            _DEFAULT_SOURCE
    )

    install(TARGETS aegisflow_kafka RUNTIME DESTINATION bin)
    message(STATUS "[AegisFlow] Kafka exporter enabled → target: aegisflow_kafka")
endif()

################################################################################
# Examples
################################################################################
if(BUILD_EXAMPLES)
    add_executable(example_live_capture examples/live_capture.c)
    target_link_libraries(example_live_capture PRIVATE aegisflow_core)
    target_compile_options(example_live_capture PRIVATE ${AEGISFLOW_WARNINGS})

    add_executable(example_pcap_replay examples/pcap_replay.c)
    target_link_libraries(example_pcap_replay PRIVATE aegisflow_core)
    target_compile_options(example_pcap_replay PRIVATE ${AEGISFLOW_WARNINGS})

    message(STATUS "[AegisFlow] Building examples")
endif()

################################################################################
# Unit Tests
################################################################################
if(BUILD_TESTS)
    enable_testing()

    set(TEST_SOURCES
        tests/test_features.c
        tests/test_flow.c
        tests/test_exporter.c
    )

    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})

        target_include_directories(${test_name}
            PRIVATE
                ${CMAKE_SOURCE_DIR}/include
                ${CMAKE_SOURCE_DIR}/tests
                ${UTHASH_INCLUDE_DIR}
        )

        target_link_libraries(${test_name} PRIVATE aegisflow_core)
        target_compile_options(${test_name} PRIVATE ${AEGISFLOW_WARNINGS})
        target_compile_definitions(${test_name}
            PRIVATE _POSIX_C_SOURCE=200809L)

        add_test(NAME ${test_name} COMMAND ${test_name})
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 30)
    endforeach()

    message(STATUS "[AegisFlow] Building tests (${CMAKE_SOURCE_DIR}/tests/)")
endif()

################################################################################
# Benchmarks
################################################################################
if(BUILD_BENCHMARKS)
    add_executable(bench_flow benchmarks/bench_flow.c)
    target_link_libraries(bench_flow PRIVATE aegisflow_core)
    target_compile_options(bench_flow PRIVATE ${AEGISFLOW_WARNINGS})
    # Enable maximum optimisation for benchmarks
    target_compile_options(bench_flow PRIVATE -O3 -march=native)
    target_compile_definitions(bench_flow
        PRIVATE
            _POSIX_C_SOURCE=200809L
            _DEFAULT_SOURCE
    )

    message(STATUS "[AegisFlow] Building benchmarks")
endif()

################################################################################
# Summary
################################################################################
message(STATUS "")
message(STATUS "══════════════════════════════════════════")
message(STATUS "  AegisFlow v${PROJECT_VERSION} configuration:")
message(STATUS "  Build type    : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C standard    : C${CMAKE_C_STANDARD}")
message(STATUS "  Tests         : ${BUILD_TESTS}")
message(STATUS "  Benchmarks    : ${BUILD_BENCHMARKS}")
message(STATUS "  Examples      : ${BUILD_EXAMPLES}")
message(STATUS "  Kafka exporter: ${AEGISFLOW_KAFKA}")
message(STATUS "  ASan/UBSan    : ${AEGISFLOW_ASAN}")
message(STATUS "══════════════════════════════════════════")
message(STATUS "")

