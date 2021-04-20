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
        void add_layer(std::size_t nodes, 
                Activation activation = Activation::None); 
        /* Set the loss function */
        void set_loss(LossFunction loss) { loss_p = loss; }
        /* Apply the model to some input */
        Vector operator()(const Vector &input) const;
        /* Compute the loss function on the difference between the result
         * of applying the model to `input` and the provided `targets`.
         */
        elem_type score(const Vector &input, const Vector &targets) const;

        /* Backpropagates on one input to compute the gradient. 
         * Assumes the right pairing between output activation and loss.
         */
        std::vector<std::pair<Matrix, Vector>> gradient(
                const Vector &input, const Vector &targets) const;
        /* Training schedule. Recieves labeled instances and number of epochs.
         * Uses stochastic gradient descent for now.
         */
        void train(const std::vector<std::pair<Vector, Vector>> instances,
                std::size_t epochs);

        /* Accessor functions to specific layers */
        const Layer &get_layer(std::size_t index) const { return layers[index]; }
        Layer &get_layer(std::size_t index) { return layers[index]; }
        /* Accessor function to loss type */
        LossFunction loss() const { return loss_p; }
    private:
        const std::size_t input_size;
        std::vector<Layer> layers;
        LossFunction loss_p;
};

} // namespace my_nn

#endif // MODEL_H
