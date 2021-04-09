/*      main.cpp 
 *
 *      main driver for the program
 */

#include <iostream>

#include "layer.h"
using namespace my_nn;


int main() {
    Layer l(5, 5);
    container input {1.0, 1.0, 1.0, 1.0, 1.0};
    auto result = l(input);
    std::cout << result[0] << " " << result[1] <<'\n';
    Layer l2(5, 5, Activation::ReLU);
    result = l2(input);
    std::cout << result[0] << " " << result[1] << '\n';
}
