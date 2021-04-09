/*      model.cpp
 *
 *      implementation file for the Model class
 */

#include "layer.h"
#include "model.h"

namespace my_nn {

void Model::addLayer(const int nodes, Activation activation) & {
    int fanin = 0;
    if (layers.size() == 0) {
        fanin = input_size;
    } else {
        fanin = layers[layers.size() - 1].nodes();
    }
    layers.push_back(Layer(fanin, nodes, activation));
}

} // namespace my_nn
