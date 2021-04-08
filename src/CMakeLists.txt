add_library(Layer layer.cpp)

add_executable(my_nn main.cpp)

target_link_libraries(my_nn PUBLIC Layer)
