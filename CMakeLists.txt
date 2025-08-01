cmake_minimum_required(VERSION 3.15)
project(video_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DUMMY_DETECTOR "Use dummy detector" OFF)
option(USE_HW_CODEC "use hardware accelerator" ON)
option(USE_TOPS_FFMPEG "use Tops ffmpeg" ON)

set(CMAKE_CXX_FLAGS -fsanitize=address ${CMAKE_CXX_FLAGS})

# Find required packages
find_package(OpenCV REQUIRED)

if (NOT USE_DUMMY_DETECTOR)
    # Find TopsInference
    find_path(TOPSINFERENCE_INCLUDE_DIR TopsInference/TopsInferRuntime.h
            PATHS /opt/tops/include $ENV{HOME}/enrigin_sdk/usr/include
            DOC "TopsInference include directory")

    find_path(TOPS_INCLUDE_DIR tops/tops_ext.h
            PATHS /opt/tops/include $ENV{HOME}/enrigin_sdk/opt/tops/include
            DOC "TopsInference include directory")

    find_library(TOPSINFERENCE_LIBRARY TopsInference
            PATHS /lib /opt/tops/lib $ENV{HOME}/enrigin_sdk/usr/lib
            DOC "TopsInference library")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
    message(STATUS "detected macOS system")
    # macOS 特定配置
    #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
    # 示例：链接 macOS 框架
    # find_library(CORE_FOUNDATION CoreFoundation)
    # target_link_libraries(your_target PRIVATE ${CORE_FOUNDATION)
    include_directories(/opt/homebrew/include)
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    message(STATUS "detected Linux system")
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    message(STATUS "detected  Windows system")
else()
	message(STATUS "Unknown：${CMAKE_SYSTEM_NAME}")
endif()

if (NOT USE_TOPS_FFMPEG)
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
   
    set(FFMPEG_LINK_LIBS PkgConfig::FFMPEG)

else()
    set(FFMPEG_INCLUDE_DIRS /home/jovyan/buildroot/include)
    set(FFMPEG_LINK_DIRS /opt/tops/lib)
    set(FFMPEG_LINK_LIBS avcodec avformat avfilter avutil swscale avdevice)
endif()

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../otl
        ${CMAKE_CURRENT_SOURCE_DIR}/include
	${TOPSINFERENCE_INCLUDE_DIR}/include
	${TOPS_INCLUDE_DIR}
	${FFMPEG_INCLUDE_DIRS}
)

link_directories(${FFMPEG_LINK_DIRS})

add_subdirectory(../otl otl.dir)

if (USE_DUMMY_DETECTOR)
    set(DETECTOR_SRCS detector_dummy.cpp)
else()
    set(DETECTOR_SRCS detector_yolo.cpp)
endif()

# Add source files
add_executable(${PROJECT_NAME}
        main.cpp
        streamer.cpp
        streamer.h
        detector.cpp
        detector.h
        device_manager.cpp
        device_manager.h
        ${DETECTOR_SRCS}
)

# Include directories
target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        ${TOPSINFERENCE_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${FFMPEG_LINK_LIBS}
        ${OpenCV_LIBS}
        otl
        ${TOPSINFERENCE_LIBRARY}
	pthread
)

# Enable C++17 features
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add install target
install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION bin
)
