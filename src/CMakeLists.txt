add_library(neural_net layer.cpp model.cpp)

add_executable(my_nn main.cpp)

target_link_libraries(my_nn PUBLIC neural_net)
