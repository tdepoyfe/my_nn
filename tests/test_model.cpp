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
        m.add_layer(5, Activation::ReLU);
        m.add_layer(3, Activation::ReLU);
        m.add_layer(1);
        auto l = m.get_layer(0);
        double x = l.weights()(0, 0);
    });
    SUCCEED();
}

/* Check the the loss function sets */
TEST(Model, ModelLoss) {
    ASSERT_NO_THROW({
        Model m(100);
        m.add_layer(100, Activation::ReLU);
        m.add_layer(50, Activation::ReLU);
        m.add_layer(1);
        m.set_loss(LossFunction::LstSq);
    });
}

/* Check that mapping works and send 0 to 0 */
TEST(Model, Model0Stable) {
    Model m(100);
    m.add_layer(100, Activation::ReLU);
    m.add_layer(50, Activation::ReLU);
    m.add_layer(1);
    m.set_loss(LossFunction::LstSq);
    Vector input = Vector::Constant(100, 0.0);
    Vector labels = Vector::Constant(1, 0.0);
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
    m.add_layer(100, Activation::ReLU);
    m.add_layer(1);
    m.set_loss(LossFunction::LstSq);
    Vector input = Vector::Constant(1, 0.5);
    Vector label = Vector::Constant(1, 0.5);
    auto output = m.score(input, label);
    auto gradient = m.gradient(input, label)[0].first(0,0);
    m.get_layer(0).weights()(0,0) += epsilon;
    auto var_output = m.score(input, label);
    auto var_grad = (var_output - output) / epsilon;
    ASSERT_NEAR(gradient, var_grad, 0.01);
}

/* Check that training reduces the error */
TEST(Model, ModelTraining) {
    // Model
    Model m(1);
    m.add_layer(10, Activation::ReLU);
    m.add_layer(1);
    m.set_loss(LossFunction::LstSq);

    // Dataset
    std::default_random_engine generator;
    std::uniform_real_distribution<elem_type> distribution_x(0.0, 1.0);
    std::normal_distribution<elem_type> distribution_noise(0.0, 0.01);
    std::vector<std::pair<Vector, Vector>> data(100);
    for (auto &instance : data) {
        auto x = distribution_x(generator);
        auto y = x*x + 1.0 + distribution_noise(generator);
        Vector input(1);
        input << x;
        Vector label(1);
        label << y;
        instance = std::pair(input, label);
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
