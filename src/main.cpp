/*      main.cpp 
 *
 *      main driver for the program
 */

#include <iostream>
#include <valarray>

#include "layer.h"
using namespace my_nn;

int main() {
    Layer l(5);
    std::valarray<double> input {1.0, 1.0, 1.0, 1.0, 1.0};
    auto result = l(input);
    std::cout << result[0] << result[1];
}
