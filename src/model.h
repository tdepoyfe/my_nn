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
        /* Constructor: need the input size to build layers. */
        Model(std::size_t input_size): input_size{input_size}, layers{} {}
        /* Add a Layer at the end of the model, connected to the previous layer */
        void addLayer(std::size_t nodes, 
                Activation activation = Activation::None); 
        /* Apply the model to some input */
        container operator()(const container &input) const;

        /* Accessor function to specific layers */
        const Layer &get_layer(std::size_t index) const { return layers[index]; }
    private:
        const std::size_t input_size;
        std::vector<Layer> layers;
};

} // namespace my_nn

#endif // MODEL_H
