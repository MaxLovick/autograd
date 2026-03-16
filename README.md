# autograd.c
 
An autograd engine written in pure C.
 
## Motivation
 
Sometimes I want to test a small neural network idea without using PyTorch, setting up a Python environment, and dealing with dependency conflicts. I built `autograd.c` so I could write self-contained programs in a single C file that compiles anywhere with a standard C compiler.
 
## What's included
 
**Tensor operations:** reshape, concatenation, broadcasting addition, scalar multiplication, element-wise multiplication, matrix multiplication, and N-dimensional convolution.
 
**Activation functions:** sigmoid, softmax, GELU, Swish, and Mish.
 
**Normalization:** Layer Norm and RMS Norm, both allow batch dimension information.
 
**Loss functions:** categorical cross-entropy (with label smoothing) and mean squared error, each supporting no batch reduction, sum, and mean reduction modes similar to Pytorch.
 
**Automatic differentiation:** a DAG-based tape that records the forward pass and computes gradients in the backwards pass. Every operation listed above has a corresponding backward implementation.
 
**Parameter initialization:** automatically selects an initialization strategy (He, Glorot, orthogonal via Householder QR) based on a BFS scan of connected nodes. Detects whether parameters feed into ReLU-like activations, softmax, recurrent connections, convolutions, or normalization layers and adjusts accordingly.
 
**Optimizers:** — SGD included, planning to add ADAM, ANO and MUON in the future.
 
**Utilities:** — a PCG32 random number generator, Gaussian sampling via Box-Muller, and a NumPy-style tensor printer with automatic summarization for large tensors.
 
## Quick start
 
A small two-layer network that learns XOR:
 
```c
#include "tensors.c"
 
int main() {
    PCGRandomNumberGenerator rng = { .state = 42, .inc = 54 };
    DirectedAcyclicGraph dag = create_DirectedAcyclicGraph();
 
    // --- Define graph nodes ---
 
    // Input: batch of 4 samples, 2 features each
    size_t input_node = add_node(&dag, OP_TYPE_INPUT, NULL);
 
    // Hidden layer: W1 [2, 8], b1 [1, 8]
    size_t W1_dims[] = {1, 1, 1, 2, 8};
    ParameterConfig W1_cfg = { .dims = W1_dims, .allow_parameter_updates = true,
        .optimization_algorithm = OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT,
        .optimizer_config = &(StochasticGradientDescentConfig){ .learning_rate = 0.1f } };
    size_t W1_node = add_node(&dag, OP_TYPE_PARAMETERS, &W1_cfg);
 
    size_t b1_dims[] = {1, 1, 1, 1, 8};
    ParameterConfig b1_cfg = { .dims = b1_dims, .allow_parameter_updates = true,
        .optimization_algorithm = OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT,
        .optimizer_config = &(StochasticGradientDescentConfig){ .learning_rate = 0.1f } };
    size_t b1_node = add_node(&dag, OP_TYPE_PARAMETERS, &b1_cfg);
 
    size_t matmul1_node = add_node(&dag, OP_TYPE_MATMUL, NULL);
    size_t add1_node = add_node(&dag, OP_TYPE_ADD, NULL);
    size_t act1_node = add_node(&dag, OP_TYPE_GELU, NULL);
 
    // Output layer: W2 [8, 1], b2 [1, 1]
    size_t W2_dims[] = {1, 1, 1, 8, 1};
    ParameterConfig W2_cfg = { .dims = W2_dims, .allow_parameter_updates = true,
        .optimization_algorithm = OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT,
        .optimizer_config = &(StochasticGradientDescentConfig){ .learning_rate = 0.1f } };
    size_t W2_node = add_node(&dag, OP_TYPE_PARAMETERS, &W2_cfg);
 
    size_t b2_dims[] = {1, 1, 1, 1, 1};
    ParameterConfig b2_cfg = { .dims = b2_dims, .allow_parameter_updates = true,
        .optimization_algorithm = OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT,
        .optimizer_config = &(StochasticGradientDescentConfig){ .learning_rate = 0.1f } };
    size_t b2_node = add_node(&dag, OP_TYPE_PARAMETERS, &b2_cfg);
 
    size_t matmul2_node = add_node(&dag, OP_TYPE_MATMUL, NULL);
    size_t add2_node = add_node(&dag, OP_TYPE_ADD, NULL);
 
    // Loss
    size_t target_node = add_node(&dag, OP_TYPE_TARGET, NULL);
    MeanSquareErrorConfig mse_cfg = { .batch_dim = 3, .reduction_type = REDUCTION_TYPE_MEAN };
    size_t loss_node = add_node(&dag, OP_TYPE_MEAN_SQUARE_ERROR, &mse_cfg);
 
    // --- Wire edges ---
    add_edge(&dag, input_node, matmul1_node);    // input × W1
    add_edge(&dag, W1_node, matmul1_node);
    add_edge(&dag, matmul1_node, add1_node);     // + b1
    add_edge(&dag, b1_node, add1_node);
    add_edge(&dag, add1_node, act1_node);        // GELU
 
    add_edge(&dag, act1_node, matmul2_node);     // hidden × W2
    add_edge(&dag, W2_node, matmul2_node);
    add_edge(&dag, matmul2_node, add2_node);     // + b2
    add_edge(&dag, b2_node, add2_node);
 
    add_edge(&dag, add2_node, loss_node);        // MSE(prediction, target)
    add_edge(&dag, target_node, loss_node);
 
    initialize_parameters(&dag, &rng);
 
    // --- XOR data ---
    size_t input_dims[] = {1, 1, 1, 4, 2};
    Tensor input_tensor = create_tensor(input_dims);
    float xor_inputs[] = {0,0, 0,1, 1,0, 1,1};
    memcpy(input_tensor.data, xor_inputs, sizeof(xor_inputs));
 
    size_t target_dims[] = {1, 1, 1, 4, 1};
    Tensor target_tensor = create_tensor(target_dims);
    float xor_targets[] = {0, 1, 1, 0};
    memcpy(target_tensor.data, xor_targets, sizeof(xor_targets));
 
    // --- Train ---
    for(int step = 0; step < 1000; step++) {
        add_input(&dag, input_node, input_tensor);
        add_input(&dag, target_node, target_tensor);
        forward(&dag);
 
        if(step % 100 == 0) {
            size_t loss_tape_idx = dag.nodes[loss_node].tape_entry_indices[dag.nodes[loss_node].total_tape_entries - 1];
            printf("step %4d  loss: %.6f\n", step, dag.tape_entries[loss_tape_idx].output_tensor.data[0]);
        }
 
        backward(&dag);
        update_parameters(&dag);
        reset_memory(&dag, false);
    }
 
    // --- Final predictions ---
    add_input(&dag, input_node, input_tensor);
    add_input(&dag, target_node, target_tensor);
    forward(&dag);
 
    size_t output_tape_idx = dag.nodes[add2_node].tape_entry_indices[dag.nodes[add2_node].total_tape_entries - 1];
    printf("\nResults:\n");
    printf("  [0, 0] -> %.4f  (target: 0)\n", dag.tape_entries[output_tape_idx].output_tensor.data[0]);
    printf("  [0, 1] -> %.4f  (target: 1)\n", dag.tape_entries[output_tape_idx].output_tensor.data[1]);
    printf("  [1, 0] -> %.4f  (target: 1)\n", dag.tape_entries[output_tape_idx].output_tensor.data[2]);
    printf("  [1, 1] -> %.4f  (target: 0)\n", dag.tape_entries[output_tape_idx].output_tensor.data[3]);
 
    return 0;
}
```
 
## Design choices
- **Tape-based autograd (similar to pytorch):** every forward operation appends to a linear tape. Backward walks the tape in reverse. Recurrent connections are supported by tracking time steps.
- **Zero external dependencies:** only the C standard library.
- **No error checking:** assume that the user has planned out the network and won't need dimension checking.  Also the networks will be small enough and won't be important enough to need memory allocation checks.  This eliminated a lot of code I didn't want to write.

## Limitations
 
- **No GPU support**
- **Small operation set** 
- **No data loading, preprocessing, or serialization**
- **No automatic mixed precision or quantization**
 
## Future improvements
 
**Performance**
- Reduce `realloc` frequency by adding size and capacity variables
- Improve matmul and convolution cache locality (tiling / blocking for better L1/L2 utilization)

**Initializers**
- Emergence Promoting Initialization (Advancing Neural Network Performance through Emergence-Promoting Initialization Scheme)

**Hyperparameter tuning**
- Write a PROTEIN like function similar to Pufferlib for hyperparameter sweeps

**Operations & losses**
- Chemical reaction operations (GenAI-Net: A Generative AI Framework for Automated Biomolecular Network Design)
- Additional non linear functions (Mining Generalizable Activation Functions)
- KL divergence loss
 
**Optimizers**
- Adam
- Muon
- ANO

**Storage & usability**
- Add functions to read and write to files so that I can use trained models in both pytorch and autograd.c
 
## Research directions
 
The area I'm most interested in exploring is using evolutionary optimizers to study how models make runtime improvements in RL environments (e.g. a game-playing agent improves its strategy based on opponent strategy). Then see if interpretability methods can identify what changes inside the network when this capability emerges,and whether that understanding can be transferred to different domains such as language processing.
