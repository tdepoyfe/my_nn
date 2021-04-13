/*      test_model.cpp
 *
 *      Tests for the Model class.
 */

#include "gtest/gtest.h"

#include "model.h"
using namespace my_nn;

/* Check that construction of the model works and that they initialize their
 * weights.
 */
TEST(Model, ModelConstruct) {
    ASSERT_NO_THROW({ 
        Model m(5);
        m.addLayer(5, Activation::ReLU);
        m.addLayer(3, Activation::ReLU);
        m.addLayer(1);
        auto l = m.get_layer(0);
        double x = l.weights()[0];
    });
    SUCCEED();
}

/* Check that mapping works and send 0 to 0 */
TEST(Model, Model0Stable) {
    Model m(100);
    m.addLayer(100, Activation::ReLU);
    m.addLayer(50, Activation::ReLU);
    m.addLayer(1);
    container input(0.0, 100);
    auto output = m(input);
    for (double x : output) {
        EXPECT_DOUBLE_EQ(x, 0.0);
    }
}
