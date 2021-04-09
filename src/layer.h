/*       layer.h
 *          Header file for the Layer class
 */

#ifndef LAYER_H
#define LAYER_H

#include <valarray>


namespace my_nn {

/* An enum to hold the type of activation function for the layer. */
enum class Activation { None, ReLU };

/* Layer
 * 
 * A class modeling a neural network layer. Holds its own weights.
 */

class Layer {
    public:
        /* Constructor; needs the previous layer size (`fanin`) to initialize
         * the weights. No activation function by default.
         */
        explicit Layer(const int fanin, const int nodes, 
                Activation activation = Activation::None);

        /* Access function to the weights. Linear access, will be deleted later */
        double const &operator[](int i) { return weights[i]; }
        /* The layer can be applied as a function to an input vector,
         * return the result. */
        std::valarray<double> operator()(std::valarray<double> const &input) const &;

        int nodes() const & { return nodes_number; }
    private:
        const int fanin;
        const int nodes_number;
        std::valarray<double> weights;
        Activation activation;
};

} // namespace my_nn

#endif // LAYER_H
