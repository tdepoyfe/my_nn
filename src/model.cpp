/*      model.cpp
 *
 *      implementation file for the Model class
 */

#include <cstdlib>
#include <stdexcept>

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

} // namespace my_nn
