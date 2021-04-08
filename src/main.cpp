/*      main.cpp 
 *
 *      main driver for the program
 */

#include <iostream>
#include <random>
#include <valarray>

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

class Layer {
    public:
        Layer(int s);
        double& operator[](int i) { return weights[i]; }
        std::valarray<double> operator()(std::valarray<double> const &input) const &;
    private:
        const int size;
        std::valarray<double> weights;
};

Layer::Layer(int s)
    : size{s}, weights(s*s)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 2 / static_cast<double>(s));
    for (auto &x: weights) {
        x = distribution(generator);
    }
}

std::valarray<double> Layer::operator()(std::valarray<double> const &input) const & {
    std::valarray<double> activations =  matmul(size, 1, size, weights, input);
    return activations;
}


int main() {
    Layer l(5);
    std::valarray<double> input {1.0, 1.0, 1.0, 1.0, 1.0};
    std::cout << l(input)[0];
}
