/*       layer.h
 *          Header file for the Layer class
 */

#ifndef LAYER_H
#define LAYER_H

#include <cstdlib>

#include "Eigen/Dense"

namespace my_nn {

using elem_type = double;
// matrix and vector with dynamically assigned dimensions. Might become
// parameters at some point.
using Matrix = Eigen::Matrix<elem_type, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<elem_type, Eigen::Dynamic, 1>;

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
        Vector operator()(const Vector &input) const;

        std::size_t nodes() const { return nodes_p; }
        std::size_t input() const { return fanin; }
        const Matrix &weights() const { return weights_p; }
        Matrix &weights() { return weights_p; }
        const Vector &bias() const { return bias_p; }
        Vector &bias() { return bias_p; }
        Activation activation() const { return activation_p; }
        
    private:
        const std::size_t fanin;
        const std::size_t nodes_p;
        Matrix weights_p;
        Vector bias_p;
        Activation activation_p;
};

} // namespace my_nn

#endif // LAYER_H
