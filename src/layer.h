/*       layer.h
 *          Header file for the Layer class
 */

#ifndef LAYER_H
#define LAYER_H

#include <valarray>

namespace my_nn {

class Layer {
    public:
        Layer(int s);
        double& operator[](int i) { return weights[i]; }
        std::valarray<double> operator()(std::valarray<double> const &input) const &;
    private:
        const int size;
        std::valarray<double> weights;
};

} // namespace my_nn

#endif // LAYER_H
