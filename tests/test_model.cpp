/*      test_model.cpp
 *
 *      Tests for the Model class.
 */

#include <random>

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
        ASSERT_NEAR(x, 0.0, 1e-100);
    }
    auto loss = m.score(input, labels);
    ASSERT_NEAR(loss, 0.0, 1e-100);
}

/* Check that the gradient procedure works by comparing with finite differences
 */
TEST(Model, ModelGradient) {
    elem_type epsilon = 0.01;
    Model m(1);
    m.addLayer(1, Activation::ReLU);
    m.addLayer(1);
    m.setLoss(LossFunction::LstSq);
    container input(0.5, 1);
    container label(0.5, 1);
    auto output = m.score(input, label);
    auto gradient = m.gradient(input, label)[0][0];
    std::vector<container> variations = { container {epsilon, 0.0}, 
                                            container {0.0, 0.0}};
    m.add_to_weights(variations);
    auto var_output = m.score(input, label);
    auto var_grad = (var_output - output) / epsilon;
    ASSERT_NEAR(gradient, var_grad, 0.01);
}

/* Check that training reduces the error */
TEST(Model, ModelTraining) {
    // Model
    Model m(1);
    m.addLayer(10, Activation::ReLU);
    m.addLayer(1);
    m.setLoss(LossFunction::LstSq);

    // Dataset
    std::default_random_engine generator;
    std::uniform_real_distribution<elem_type> distribution_x(0.0, 1.0);
    std::normal_distribution<elem_type> distribution_noise(0.0, 0.01);
    std::vector<std::pair<container, container>> data(100);
    for (auto &instance : data) {
        container x { distribution_x(generator) };
        container y { x*x + 1.0 + distribution_noise(generator) };
        instance = std::pair(x, y);
    }

    // Initial prediction
    auto initial_loss = 0.0;
    for (auto &instance : data) {
        auto x = instance.first;
        auto y = instance.second;
        initial_loss += m.score(x, y);
    }
    initial_loss /= data.size();

    // Training
    m.train(data, 10);

    // Post-training prediction
    auto final_loss = 0.0;
    for (auto &instance : data) {
        auto x = instance.first;
        auto y = instance.second;
        final_loss += m.score(x, y);
    }
    final_loss /= data.size();

    // the loss should have decreased
    auto compare = [] (elem_type a, elem_type b) { return a < b; };
    EXPECT_PRED2(compare, final_loss, initial_loss);
}
