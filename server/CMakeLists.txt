cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(server)

#set(CMAKE_PREFIX_PATH /home/max/libtorch)
#set(Torch_DIR "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")
#set(CMAKE_PREFIX_PATH "python3 -c 'import torch; print(torch.utils.cmake_prefix_path)'")
#find_package(Torch REQUIRED)
#include_directories(${TORCH_INCLUDE_DIRS})
add_executable(server main.cpp
        ImageClassificaion.h
        ImageClassification.cpp
        WebServer.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set( LIBS_TO_LINK
        ${Boost_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        -lboost_system
        -lboost_thread
        -lpthread
)
target_link_libraries(server
        ${LIBS_TO_LINK}
        ${TORCH_LIBRARIES}
)
set_property(TARGET server PROPERTY CXX_STANDARD 17)
