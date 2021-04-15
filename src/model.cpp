/*      model.cpp
 *
 *      implementation file for the Model class
 */

#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "layer.h"
#include "model.h"

namespace my_nn {

void Model::addLayer(std::size_t nodes, Activation activation) {
    std::size_t fanin = 0;
    if (layers.size() == 0) {
        fanin = input_size;
    } else {
        fanin = layers[layers.size() - 1].nodes();
    }
    layers.push_back(Layer(fanin, nodes, activation));
}

container Model::operator()(const container &input) const {
    container scratch = input;
    for (const Layer &layer : layers) {
        scratch = layer(scratch);
    }
    return scratch;
}

elem_type Model::score(const container &inputs, const container &targets) const {
    auto results = operator()(inputs);
    switch (loss_p) {
        case LossFunction::LstSq:
            return std::pow(results - targets, 2).sum();
        case LossFunction::LogLoss:
            return (targets * std::log(results) + 
                    (1-targets) * std::log(1-results)).sum();
        default:
            throw std::invalid_argument("No loss function set");
    }
}

std::size_t Model::node_number() const {
    std::size_t number = 0;
    for (auto &layer : layers) {
        number += layer.nodes();
    }
    return number;
}

std::vector<container> Model::gradient(
        const container &input, const container & targets) const
{
    // this will store the result; for each layer, it is a matrix with
    // as many lines as nodes and as many columns as inputs + 1 for the bias
    // which will be the last column.
    std::vector<container> gradients(layers.size());
    // We need to store the activations of each node
    std::vector<container> activations(layers.size());
    
    // forward pass
    auto scratch = input; // stores the result at the current layer
    for (int i = 0; i < layers.size(); i++) {
        auto &layer = layers[i];
        // this part initialize the gradient with each line equal to the input
        // and a 1.0 in the last column for the bias input node.
        auto &grad = gradients[i];
        grad = container(layer.weights().size() + layer.bias().size());
        for (int j = 0; j < layer.nodes(); j++) {
            auto fanin = layer.input() + 1; // add 1 for the bias
            grad[ std::slice(j*fanin, fanin - 1, 1) ] = scratch;
            grad[ (j+1)*fanin - 1 ] = 1.0;
        }

        // compute the result of the layer.
        auto &acts = activations[i];
        acts = layer.mult(scratch); // no activation function
        switch (layer.activation()) {  // apply the activation function
                                        // then store the derivative in acts
            case Activation::None:
                scratch = acts;
                acts = acts.apply([](elem_type x) { return 1.0; });
                break;
            case Activation::ReLU:
                scratch = acts.apply(ReLU);
                acts = acts.apply(der_ReLU);
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
    for (int i = layers.size() - 2; i >= 0; i--) {
        auto &layer = layers[i];
        // compute the deltas by using the transpose operation and the derivative
        // of the activation function
        auto &delt = activations[i];
        scratch = layer.transp(activations[i+1]);
        delt *= scratch;

        auto &grad = gradients[i];
        for (int j = 0; j < layer.nodes(); j++) {
            auto fanin = layer.input() + 1; // add 1 for the bias
            for (int k = j*fanin; k < fanin; k++) {
                grad[k] *= delt[j];
            }
        }
    }
    
    return gradients;
}

void Model::add_to_weights(std::vector<container> variations) {
    for (int i = 0; i < layers.size(); i++) {
        auto layer = layers[i];
        auto variation = variations[i];
        auto m = layer.input();
        for (int j = 0; j < layer.weights().size(); j++) {
            layer.weights()[j] += variation[j + j / m];
        }
        for (int j = 0; j< layer.bias().size(); j++) {
            layer.bias()[j] += variation[(j+1) * (m+1)];
        }
    }
}

} // namespace my_nn
