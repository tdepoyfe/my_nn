/*       layer.h
 *          Header file for the Layer class
 */

#ifndef LAYER_H
#define LAYER_H

#include <valarray>

namespace my_nn {

enum class Activation { None, ReLU };

class Layer {
    public:
        Layer(const int fanin, const int nodes, Activation activation = Activation::None);
        double& operator[](int i) { return weights[i]; }
        std::valarray<double> operator()(std::valarray<double> const &input) const &;
    private:
        const int fanin;
        const int nodes;
        std::valarray<double> weights;
        Activation activation;
};

} // namespace my_nn

#endif // LAYER_H
