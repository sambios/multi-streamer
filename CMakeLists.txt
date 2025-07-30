cmake_minimum_required(VERSION 3.15)
project(video_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../otl
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenCV REQUIRED)

# Find FFmpeg using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavfilter
    libavutil
    libswscale
    libavdevice
)

add_subdirectory(../otl otl.dir)

# Add source files
add_executable(${PROJECT_NAME}
    main.cpp
    streamer.cpp
    streamer.h
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    PkgConfig::FFMPEG
    ${OpenCV_LIBS}
    Threads::Threads
        otl
)

# Enable C++17 features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add install target
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)
