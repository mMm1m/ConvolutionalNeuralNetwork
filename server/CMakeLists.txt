cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(image_classification)

set(CMAKE_PREFIX_PATH /home/max/libtorch)
find_package(Torch REQUIRED)
add_executable(image_classification main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set( LIBS_TO_LINK
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        -lboost_system
        -lboost_thread
        -lpthread
)

target_link_libraries(image_classification
        ${TORCH_LIBRARIES}
        ${LIBS_TO_LINK}
)
set_property(TARGET image_classification PROPERTY CXX_STANDARD 17)
