/*      model.cpp
 *
 *      implementation file for the Model class
 */

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "model.h"

namespace my_nn {

void Model::add_layer(std::size_t nodes, Activation activation) {
    std::size_t fanin = 0;
    if (layers.size() == 0) {
        fanin = input_size;
    } else {
        fanin = layers[layers.size() - 1].nodes();
    }
    layers.push_back(Layer(fanin, nodes, activation));
}

Vector Model::operator()(const Vector &input) const {
    auto scratch = input;
    for (const Layer &layer : layers) {
        scratch = layer(scratch);
    }
    return scratch;
}

elem_type Model::score(const Vector &inputs, const Vector &targets) const {
    auto results = operator()(inputs);
    switch (loss_p) {
        case LossFunction::LstSq:
            return (results - targets).squaredNorm();
        case LossFunction::LogLoss:
            return (targets.array() * results.array().log() +
                    (1 - targets.array()) * (1 - results.array()).log()).sum();
        default:
            throw std::invalid_argument("No loss function set");
    }
}

std::vector<std::pair<Matrix, Vector>> Model::gradient(
        const Vector &input, const Vector &targets) const
{
    // this will store the result
    std::vector<std::pair<Matrix, Vector>> gradients(layers.size());
    // We need to store the activations of each node, then the error at each node.
    std::vector<Vector> activations(layers.size());
    
    // forward pass
    auto scratch = input; // stores the result at the current layer
    for (int i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];
        // this part initialize the gradient with each line equal to the input
        // while the bias terms don't need to be initialized. (they are 
        // implicitly equal to 1
        auto &grad = gradients[i];
        grad.first = Matrix(layer.weights().rows(), layer.weights().cols());
        for (int j = 0; j < layer.nodes(); j++) {
            grad.first.row(j) = scratch.transpose();
        }

        // compute the result of the layer.
        auto &acts = activations[i];
        acts = layer.weights() * scratch + layer.bias(); // no activation function
        switch (layer.activation()) {  // apply the activation function
                                        // then store the derivative in acts
            case Activation::None:
                scratch = acts;
                acts = acts.array().unaryExpr([](elem_type x) { return 1.0; });
                break;
            case Activation::ReLU:
                scratch = acts.array().unaryExpr(std::ref(ReLU));
                acts = acts.array().unaryExpr(std::ref(der_ReLU));
                break;
            default:
                throw std::invalid_argument("No activation set");
        }
    }

    // reverse pass
    // initialize the deltas for the last node. this assumes the right pairing 
    // of loss function and last layer activation function.
    // the deltas are stored in activations
    activations[layers.size()-1] = scratch - targets; // scratch containes the
                                                   // result of the neural net

    // compute the deltas by using the transpose operation and the derivative
    // of the activation function stored in act
    for (int i = layers.size() - 2; i >= 0; i--) {
        auto &delt = activations[i];
        scratch = layers[i+1].weights().transpose() * activations[i+1];
        delt = delt.array() * scratch.array();
    }

    // collects the gradients
    for (int i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];
        auto &delt = activations[i];
        auto &grad = gradients[i];
        for (int j = 0; j < layer.input(); j++) {
            grad.first.col(j) = grad.first.col(j).array() * delt.array();
        }
        grad.second = delt;
    }
    
    return gradients;
}
void Model::train(const std::vector<std::pair<Vector, Vector>> instances,
            std::size_t epochs) {
    auto inst_number = instances.size();
    std::default_random_engine generator;
    std::uniform_int_distribution<std::size_t> distribution(0, inst_number-1);
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < inst_number; j++) {
            auto index = distribution(generator);
            auto input = instances[index].first;
            auto labels = instances[index].second;
            auto grad = gradient(input, labels);
            for (int k = 0; k < layers.size(); k++) {
                layers[k].weights() -= grad[k].first;
                layers[k].bias() -= grad[k].second;
            }
        }
    }
}

} // namespace my_nn
