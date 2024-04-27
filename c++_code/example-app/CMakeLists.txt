cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

#set(CMAKE_PREFIX_PATH /home/max/libtorch)
find_package(Torch REQUIRED)
add_executable(example-app example_app.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)
