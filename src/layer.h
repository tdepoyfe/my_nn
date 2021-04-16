/*       layer.h
 *          Header file for the Layer class
 */

#ifndef LAYER_H
#define LAYER_H

#include <cstdlib>

#include "Eigen/Dense"

namespace my_nn {

using elem_type = double;
// a matrix with dynamically assigned dimensions.
using Matr = Eigen::Matrix<elem_type, Eigen::Dynamic, Eigen::Dynamic>;
using Vect = Eigen::Matrix<elem_type, Eigen::Dynamic, 1>;

/* An enum to hold the type of activation function for the layer. */
enum class Activation { None, ReLU };

/* the ReLU function */
elem_type ReLU(elem_type x);
elem_type der_ReLU(elem_type x);

/* Layer
 * 
 * A class modeling a neural network layer. Holds its own weights.
 */
class Layer {
    public:
        /* Constructor; needs the previous layer size (`fanin`) to initialize
         * the weights. No activation function by default.
         */
        Layer(std::size_t fanin, std::size_t nodes, 
                Activation activation = Activation::None);

        /* The layer can be applied as a function to an input vector,
         * return the result. */
        Vect operator()(const Vect &input) const;

        auto nodes() const { return nodes_p; }
        auto input() const { return fanin; }
        const auto &weights() const { return weights_p; }
        auto &weights() { return weights_p; }
        const auto &bias() const { return bias_p; }
        auto &bias() { return bias_p; }
        auto activation() const { return activation_p; }
        
    private:
        const std::size_t fanin;
        const std::size_t nodes_p;
        Matr weights_p;
        Vect bias_p;
        Activation activation_p;
};

} // namespace my_nn

#endif // LAYER_H
