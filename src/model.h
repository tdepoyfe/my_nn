/*      model.h
 *
 *      header file for the Model class
 */

#ifndef MODEL_H
#define MODEL_H

#include <vector>

#include "layer.h"

namespace my_nn {

class Model {
    public:
        explicit Model(const int input_size): input_size{input_size}, layers{} {}
        void addLayer(const int nodes, Activation activation) &;
    private:
        const int input_size;
        std::vector<Layer> layers;
};

} // namespace my_nn

#endif // MODEL_H
