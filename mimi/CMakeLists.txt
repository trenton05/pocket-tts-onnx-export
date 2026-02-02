# Copyright (C) 2016 Verizon. All Rights Reserved.

cmake_minimum_required(VERSION 2.8)
project(mimi)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libs/libonnxruntime.so)

add_executable(mimi mimi.cpp)

target_include_directories(mimi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime )
target_link_libraries(mimi PRIVATE onnxruntime)