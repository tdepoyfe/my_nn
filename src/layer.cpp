/*      layer.cpp
 *
 *      Implementation for the layer class
 */

#include <random>
#include <valarray>

#include "layer.h"

namespace my_nn {

/* A is of size (n, l), B of size (l, m), result of size (n, m) */
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

Layer::Layer(const int fanin, const int nodes)
    : fanin{fanin}, nodes{nodes}, weights(fanin*nodes)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(
            0.0, 2 / static_cast<double>(fanin)
            );
    for (auto &x: weights) {
        x = distribution(generator);
    }
}

std::valarray<double> Layer::operator()(
        std::valarray<double> const &input
        ) const & 
{
    std::valarray<double> activations =  matmul(nodes, 1, fanin, weights, input);
    auto ReLU = [](double x) { return x > 0.0 ? x : 0.0; };
    return activations.apply(ReLU);
}

}   // namespace my_nn
