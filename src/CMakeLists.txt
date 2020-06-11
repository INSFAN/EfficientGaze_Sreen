cmake_minimum_required(VERSION 2.8)

project(EfficientGaze)
set(CMAKE_CXX_FLAGS -lboost_system)
set(CMAKE_CXX_FLAGS -lboost_filesystem)

add_definitions(-std=c++11)
add_definitions("-Wall")

set(OpenCV_DIR "/usr/local/opencv/")
# set(ncnn_DIR "/usr/local/opencv/")
# Boost, for reading configuration file
find_package(Boost 1.55 COMPONENTS system filesystem  program_options REQUIRED)
# find_package(Boost 1.55.0 REQUIRED)
set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR} ${Boost_INCLUDE_DIR}/boost)
# set(Boost_INCLUDE_DIRS "/usr/include/boost/")

# set(Boost_LIBRARY_DIRS "/usr/lib/x86_64-linux-gnu/")

# find_package(Boost 1.58.0 REQUIRED COMPONENTS system filesystem)

# include_directories(... ${Boost_INCLUDE_DIRS})

# link_directories(... ${Boost_LIBRARY_DIRS})

# target_link_libraries(... ${Boost_LIBRARIES})

if(Boost_FOUND)
    # include_directories(${Boost_INCLUDE_DIRS})
    MESSAGE( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
    MESSAGE( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
    MESSAGE( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")
endif()


if(True)
    find_package(OpenMP)
    if(NOT TARGET OpenMP::OpenMP_CXX AND (OpenMP_CXX_FOUND OR OPENMP_FOUND))
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

include_directories(${OpenCV_DIR}/include
                    ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/include/ncnn
                    ${Boost_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include)

link_directories(${OpenCV_DIR}/lib/
                 ${CMAKE_CURRENT_SOURCE_DIR}/../ncnn/lib
                 ${Boost_LIBRARY_DIRS}
                )
# include_directories(./ ./include /usr/local/cuda/include ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CLM_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})
# link_directories(./ ./build/lib /usr/lib /usr/local/cuda/lib64 ${Boost_LIBRARY_DIRS} ${CLM_LIBRARY_DIRS} ${Caffe_LIBRARY_DIRS})

# file(GLOB SOURCE "../src/*.cpp")
# file(GLOB HEADERS "./include/*.hpp")

# compile opengaze library
# add_library(EfficientGaze SHARED ${SOURCE} ${HEADERS})
# set_target_properties(opengaze PROPERTIES VERSION ${PROJECT_VERSION})


set(OpenCV_LIBS opencv_core 
                opencv_highgui 
                opencv_imgproc 
                opencv_imgcodecs 
                opencv_videoio
                opencv_calib3d)

# set(SRC
#         main.cpp
#         FaceDetector.h
#         FaceDetector.cpp
#         )
set(SRC
        # face_landmks_detect_main.cpp
        # FaceDetector.hpp
        # FaceDetector.cpp
        # FaceLandmksDetector.h
        # FaceLandmksDetector.cpp
        # SOURCE
        # HEADERS
        face_detector.cpp
        GazeVisualization.cpp
        gaze_estimator.cpp
        gaze_predictor.cpp
        input_handler.cpp
        normalizer.cpp
        opengaze.cpp
        personal_calibrator.cpp
        )

# lboost_system,  lboost_filesystem


add_executable(EfficientGaze ${SRC})
target_link_libraries(EfficientGaze ncnn ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(EfficientGaze X11)
if(OpenMP_CXX_FOUND)
    if(NCNN_CMAKE_VERBOSE)
        message("Building with OpenMP")
    endif()
    target_link_libraries(EfficientGaze PUBLIC OpenMP::OpenMP_CXX)
endif()
