/*      model.cpp
 *
 *      implementation file for the Model class
 */

#include <cstdlib>

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

} // namespace my_nn
