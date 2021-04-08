/* test_layer.cpp
 *
 * unit tests for Layer class
 */
#include "valarray"

#include "gtest/gtest.h"

#include "layer.h"
using namespace my_nn;

/* Check that construction of the layers work and that they initialize their
 * weights.
 */
TEST(Layer, LayerConstruct) {
    ASSERT_NO_THROW({ 
        Layer l(1, 1);
        double x = l[0];
    });
    SUCCEED();
}

/* Check that input 0 is mapped to output 0 by the layers.
 */
TEST(Layer, Layer0Stable) {
    Layer l(100, 100, Activation::ReLU);
    std::valarray<double> input(0.0, 100);
    auto output = l(input);
    for (double x : output) {
        EXPECT_DOUBLE_EQ(x, 0.0);
    }
}
    
