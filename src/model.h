/*      model.h
 *
 *      header file for the Model class
 */

#ifndef MODEL_H
#define MODEL_H

#include <cstdlib>
#include <vector>

#include "layer.h"

namespace my_nn {

class Model {
    public:
        Model(std::size_t input_size): input_size{input_size}, layers{} {}
        void addLayer(std::size_t nodes, Activation activation); 
    private:
        const std::size_t input_size;
        std::vector<Layer> layers;
};

} // namespace my_nn

#endif // MODEL_H
