/*      main.cpp 
 *
 *      main driver for the program
 */

#include <iostream>

#include "model.h"
using namespace my_nn;


int main() {
    Model m(5);
    m.addLayer(5, Activation::ReLU);
    m.addLayer(3, Activation::ReLU);
    m.addLayer(1);

    container input {1.0, 1.0, 1.0, 1.0, 1.0};
    auto result = m(input);
    std::cout << result[0] <<'\n';
}
