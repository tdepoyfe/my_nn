/*      layer.cpp
 *
 *      Implementation for the layer class
 */

#include <cstdlib>
#include <random>
#include <stdexcept>
#include <valarray>

#include "layer.h"

namespace my_nn {

/* valarray matrix multiplication. Placeholder for a better implementation later.
 * A is of size (n, l), B of size (l, m), result of size (n, m) */
std::valarray<double> matmul(int n, int m, int l,
        const std::valarray<double> &A, 
        const std::valarray<double> &B)
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

/* valarray matrix transpose multiplication AT * B = C.
 * A is of size (l, n), B of size (l, m), result of size (n, m) */
std::valarray<double> matmultransp(int n, int m, int l,
        const std::valarray<double> &A, 
        const std::valarray<double> &B)
{
    std::valarray<double> C(0.0, n * l);
    for (int k = 0; k < l; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                C[i*m+j] += A[k*n+i] * B[k*m+j];
            }
        }
    }
    return C;
}

elem_type ReLU(elem_type x) { return x > 0.0 ? x : 0.0; }
elem_type der_ReLU(elem_type x) { return x > 0.0 ? 1.0 : 0.0; }

/* Constructor with He initialization of the weights
 */
Layer::Layer(std::size_t fanin, std::size_t nodes, Activation activation)
    : fanin{fanin}, nodes_p{nodes}, 
    weights_p(fanin*nodes), bias_p(0.0, nodes), activation_p{activation}
{
    std::default_random_engine generator;
    std::normal_distribution<elem_type> distribution(
            0.0, 2 / static_cast<elem_type>(fanin)
            );
    for (auto &x: weights_p) {
        x = distribution(generator);
    }
}

container Layer::mult(const container &input) const{
    return matmul(nodes_p, 1, fanin, weights_p, input) + bias_p;
}

container Layer::transp(const container &input) const {
    container result(fanin + 1);
    result[ std::slice(0, result.size()-1, 1) ] 
        = matmultransp(fanin, 1, nodes_p, weights_p, input);
    result[result.size() - 1] = (bias_p * input).sum();
    return result;
}


/* Application of the layer is Matrix multiplication of the input vector by the
 * weights followed by the activation function term by term.
 */
container Layer::operator()(const container &input) const {
    container act = mult(input);

    switch (activation_p) {
        case Activation::None:
            break;
        case Activation::ReLU:
            act = act.apply(ReLU);
            break;
        default:
            throw std::invalid_argument("No activation set");
    }
    return act;

}

}   // namespace my_nn
