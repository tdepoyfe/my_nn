/*       layer.h
 *          Header file for the Layer class
 */

#ifndef LAYER_H
#define LAYER_H

#include <cstdlib>
#include <valarray>

namespace my_nn {

using elem_type = double;
using container = std::valarray<elem_type>;

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
        container operator()(const container &input) const;
        /* Apply the layer multiplication */
        container mult(const container &input) const;
        /* Returns the transpose of the weights and biases */
        container transp(const container &input) const;

        auto nodes() const { return nodes_p; }
        auto input() const { return fanin; }
        const auto &weights() const { return weights_p; }
        const auto &bias() const { return bias_p; }
        auto activation() const { return activation_p; }
        
    private:
        const std::size_t fanin;
        const std::size_t nodes_p;
        container weights_p;
        container bias_p;
        Activation activation_p;
};

} // namespace my_nn

#endif // LAYER_H
