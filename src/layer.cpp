/*      layer.cpp
 *
 *      Implementation for the layer class
 */

#include <random>
#include <valarray>

#include "layer.h"

namespace my_nn {

/* valarray matrix multiplication. Placeholder for a better implementation later.
 * A is of size (n, l), B of size (l, m), result of size (n, m) */
std::valarray<double> matmul(int n, int m, int l,
        std::valarray<double> const &A, 
        std::valarray<double> const &B)
{
    std::valarray<double> C(0.0, n * l);
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < l; k++) {
            for (int j = 0; j < m; j++) {
                C[i*m+j] += A[i*l+k] * B[k*m+j];
            }
        }
    }
    return C;
}

/* Constructor with He initialization of the weights
 */
Layer::Layer(const int fanin, const int nodes, Activation activation)
    : fanin{fanin}, nodes{nodes}, weights(fanin*nodes), activation{activation}
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(
            0.0, 2 / static_cast<double>(fanin)
            );
    for (auto &x: weights) {
        x = distribution(generator);
    }
}

/* the ReLU function */
double relu(double x) { return x > 0.0 ? x : 0.0; }

/* Application of the layer is Matrix multiplication of the input vector by the
 * weights followed by the activation function term by term.
 */
std::valarray<double> Layer::operator()(
        std::valarray<double> const &input
        ) const & 
{
    std::valarray<double> act =  matmul(nodes, 1, fanin, weights, input);

    switch (activation) {
        case Activation::None:
            return act;
        case Activation::ReLU:
            return act.apply(relu);
    }
}

}   // namespace my_nn
