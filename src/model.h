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

enum class LossFunction {
    Unset, LstSq, LogLoss
};

class Model {
    public:
        /* Constructor: need the input size to build layers. */
        Model(std::size_t input_size): 
            input_size{input_size}, layers{}, loss_p{LossFunction::Unset} {}
        /* Add a Layer at the end of the model, connected to the previous layer */
        void addLayer(std::size_t nodes, 
                Activation activation = Activation::None); 
        /* Apply the model to some input */
        container operator()(const container &input) const;
        /* Compute the loss function on the difference between the result
         * of applying the model to `input` and the provided `targets`.
         * The loss function will be deduced from the last layer.
         */
        elem_type score(const container &input, const container &targets) const;

        /* Accessor function to specific layers */
        const Layer &get_layer(std::size_t index) const { return layers[index]; }
        LossFunction loss() const { return loss_p; }
    private:
        const std::size_t input_size;
        std::vector<Layer> layers;
        LossFunction loss_p;
};

} // namespace my_nn

#endif // MODEL_H
