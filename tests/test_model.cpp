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

/* Check the the loss function sets */
TEST(Model, ModelLoss) {
    ASSERT_NO_THROW({
        Model m(100);
        m.addLayer(100, Activation::ReLU);
        m.addLayer(50, Activation::ReLU);
        m.addLayer(1);
        m.setLoss(LossFunction::LstSq);
    });
}

/* Check that mapping works and send 0 to 0 */
TEST(Model, Model0Stable) {
    Model m(100);
    m.addLayer(100, Activation::ReLU);
    m.addLayer(50, Activation::ReLU);
    m.addLayer(1);
    m.setLoss(LossFunction::LstSq);
    container input(0.0, 100);
    container labels {0.0};
    auto output = m(input);
    for (double x : output) {
        EXPECT_DOUBLE_EQ(x, 0.0);
    }
    auto loss = m.score(input, labels);
    EXPECT_DOUBLE_EQ(loss, 0.0);
}
