/*      main.cpp 
 *
 *      main driver for the program
 */

#include <iostream>
#include <random>
#include <vector>

class Layer {
    public:
        Layer(int s);
        double& operator[](int i) { return weights[i]; }
    private:
        int size;
        std::vector<double> weights;
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

int main() {
    Layer l(5);
    std::cout << l[0] << l[1];
}
