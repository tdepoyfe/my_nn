# get the GoogleTest dependency
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_layer test_layer.cpp)

target_link_libraries(test_layer neural_net)
target_link_libraries(test_layer gtest_main)

add_executable(test_model test_model.cpp)

target_link_libraries(test_model neural_net)
target_link_libraries(test_model gtest_main)

include(GoogleTest)
gtest_discover_tests(test_layer)
gtest_discover_tests(test_model)
