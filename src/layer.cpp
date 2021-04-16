/*      layer.cpp
 *
 *      Implementation for the layer class
 */

#include <cstdlib>
#include <random>
#include <stdexcept>

#include "Eigen/Dense"

#include "layer.h"

namespace my_nn {

elem_type ReLU(elem_type x) { return x > 0.0 ? x : 0.0; }
elem_type der_ReLU(elem_type x) { return x > 0.0 ? 1.0 : 0.0; }

/* Constructor with He initialization of the weights
 */
Layer::Layer(std::size_t fanin, std::size_t nodes, Activation activation)
    : fanin{fanin}, nodes_p{nodes}, 
    weights_p(nodes, fanin), bias_p(nodes), activation_p{activation}
{
    std::default_random_engine generator;
    std::normal_distribution<elem_type> distribution(
            0.0, 2 / static_cast<elem_type>(fanin)
            );
    for (auto &x: weights_p.reshaped()) {
        x = distribution(generator);
    }
    for (auto &x : bias_p) {
        x = 0.0;
    }
}

/* Application of the layer is Matrix multiplication of the input vector by the
 * weights followed by the activation function term by term.
 */
Vect Layer::operator()(const Vect &input) const {
    Vect act = weights_p * input + bias_p;

    switch (activation_p) {
        case Activation::None:
            break;
        case Activation::ReLU:
            act = act.array().unaryExpr(std::ref(ReLU));
            break;
        default:
            throw std::invalid_argument("No activation set");
    }
    return act;
}

}   // namespace my_nn
