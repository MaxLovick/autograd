# autograd.c

`autograd.c` is an automatic differentiation graph similar to PyTorch written in C using only the standard library. It only uses the most common standard library functions so there is little concern about whether it will be supported on most compilers.

There are no dependencies beyond `libc` and `libm`, no build system, and no headers to install so you can drop the one file into a project, write a `main`, and compile.

## Features
- **Reverse-mode automatic differentiation** over an explicit operation graph, so any composition of the built-in operators is differentiable with no extra code.
- **A broad operator set**: matrix multiply, N-dimensional convolution, elementwise and scalar multiply, add, concatenate, resize, a family of activations (sigmoid, GELU, Swish, Mish, plus GELU-sine and GELU-sinc from *Mining Generalizable Activation Functions* and Mish-sine and Mish-sinc), normalization (softmax, LayerNorm, RMSNorm), and three losses (categorical cross-entropy, mean squared error, KL divergence).
- **Optimizers**: SGD, Adam, Muon (with Newton–Schulz orthogonalization), and ANO.
- **Regularization**: L1 and L2 weight penalties applied at the start of `optimize_weights`, plus three stochastic graph operators with full forward and backward passes: Dropout, DropPath (per-sample stochastic depth), and Zoneout.
- **Initialization** picks He / Xavier / softmax-scaled variance based on downstream activation functions, applies orthogonal initialization to matmul weights, and offers an optional depth-based "emergence-promoting" rescaling. This should lead to more stable early training.

## Design

The library separates the structure of the graph from its execution using a tape-based execution model.

A `Node` holds an operation type, its configuration if needed, edges to other nodes, and the indices of its tape entries. The collection of nodes is a directed acyclic graph that you build once. A `TapeEntry` array is the  execution of a node: it owns the `output_tensor` produced on the forward pass and the `gradient_tensor` accumulated on the backward pass.

A `forward(dag)` call advances an internal time step, walks the tape executing every node whose inputs are ready, and records outputs. `backward(dag)` then walks the tape in reverse, calling the gradient function for each operation and accumulating the gradients. Because gradients accumulate with `+=`, weight sharing (the same parameter node feeding several operations, or the same cell unrolled across timesteps) is handled correctly with no special bookkeeping.

`Tensor` holds a flat `float*` buffer, a `dims[TENSOR_DIMS]` shape, and an element count. `TENSOR_DIMS` is `5` by default, but every operation is written against that constant, so you can change it in one place if your project needs a different rank. Tensors are stored row-major.

## Demo

```sh
gcc -std=c11 -O2 autograd.c -lm -o your_program
```

Any C11 compiler works; `-lm` links the math functions. The file compiles cleanly under `-Wall -Wextra`.

## Quick start

A two-layer MLP — two linear layers with broadcast biases, a GELU nonlinearity, and dropout in between — trained against an MSE target. It also shows two optimizers in one graph (Muon on the weight matrices, Adam on the biases), L2 on the weights, and both initializers.

```c
#include /* the contents of autograd.c, or compile it as your translation unit */

#define IN      16
#define HIDDEN  32
#define OUT     4
#define BATCH   8

/* Muon + L2, for the weight matrices */
static OperationConfiguration muon_weight(void) {
    OperationConfiguration c;
    c.parameters.allow_parameter_updates = true;
    c.parameters.l1_strength = 0.0f;
    c.parameters.l2_strength = 1e-4f;
    c.parameters.optimizer.type = OPTIMIZER_TYPE_MUON;
    c.parameters.optimizer.config.muon.learning_rate = 2e-2f;
    c.parameters.optimizer.config.muon.momentum = 0.95f;
    c.parameters.optimizer.config.muon.newton_schulz_steps = 5;
    c.parameters.optimizer.config.muon.nesterov = true;
    c.parameters.optimizer.config.muon.momentum_buffer.data = NULL;  /* lazily allocated */
    return c;
}

/* Adam, no decay, for the biases */
static OperationConfiguration adam_bias(void) {
    OperationConfiguration c;
    c.parameters.allow_parameter_updates = true;
    c.parameters.l1_strength = 0.0f;
    c.parameters.l2_strength = 0.0f;
    c.parameters.optimizer.type = OPTIMIZER_TYPE_ADAM;
    c.parameters.optimizer.config.adam.learning_rate = 1e-3f;
    c.parameters.optimizer.config.adam.beta1 = 0.9f;
    c.parameters.optimizer.config.adam.beta2 = 0.999f;
    c.parameters.optimizer.config.adam.epsilon = 1e-8f;
    c.parameters.optimizer.config.adam.time_step = 0;
    c.parameters.optimizer.config.adam.first_moment.data = NULL;
    c.parameters.optimizer.config.adam.second_moment.data = NULL;
    return c;
}

int main(void) {
    DirectedAcyclicGraph dag = create_directed_acyclic_graph();

    /* --- structure: x -> (W1,b1) -> GELU -> Dropout -> (W2,b2) -> MSE --- */
    size_t x = add_node(&dag, OPERATION_TYPE_INPUT, NULL);

    OperationConfiguration w1c = muon_weight();
    OperationConfiguration b1c = adam_bias();
    OperationConfiguration w2c = muon_weight();
    OperationConfiguration b2c = adam_bias();
    size_t W1 = add_node(&dag, OPERATION_TYPE_PARAMETERS, &w1c);
    size_t b1 = add_node(&dag, OPERATION_TYPE_PARAMETERS, &b1c);
    size_t W2 = add_node(&dag, OPERATION_TYPE_PARAMETERS, &w2c);
    size_t b2 = add_node(&dag, OPERATION_TYPE_PARAMETERS, &b2c);

    size_t mm1 = add_node(&dag, OPERATION_TYPE_MATRIX_MULTIPLICATION, NULL);
    add_edge(&dag, x, mm1);     /* {1,1,1,BATCH,IN} */
    add_edge(&dag, W1, mm1);    /* {1,1,1,IN,HIDDEN} */

    size_t add1 = add_node(&dag, OPERATION_TYPE_ADD, NULL);
    add_edge(&dag, mm1, add1);
    add_edge(&dag, b1, add1);   /* {1,1,1,1,HIDDEN}, broadcasts over the batch */

    size_t act = add_node(&dag, OPERATION_TYPE_GELU, NULL);
    add_edge(&dag, add1, act);

    OperationConfiguration drop_config;
    drop_config.stochastic_regularization.probability = 0.1f;
    drop_config.stochastic_regularization.sample_dimension = 0;  /* unused by dropout */
    drop_config.stochastic_regularization.base_seed = 0x1234ABCDULL;
    size_t drop = add_node(&dag, OPERATION_TYPE_DROPOUT, &drop_config);
    add_edge(&dag, act, drop);

    size_t mm2 = add_node(&dag, OPERATION_TYPE_MATRIX_MULTIPLICATION, NULL);
    add_edge(&dag, drop, mm2);
    add_edge(&dag, W2, mm2);    /* {1,1,1,HIDDEN,OUT} */

    size_t add2 = add_node(&dag, OPERATION_TYPE_ADD, NULL);
    add_edge(&dag, mm2, add2);
    add_edge(&dag, b2, add2);   /* {1,1,1,1,OUT} */

    size_t target = add_node(&dag, OPERATION_TYPE_INPUT, NULL);
    size_t loss = add_node(&dag, OPERATION_TYPE_MEAN_SQUARE_ERROR, NULL);
    add_edge(&dag, add2, loss);
    add_edge(&dag, target, loss);

    /* --- parameter storage + shapes (last two dims are the matrix) --- */
    size_t w1_dims[TENSOR_DIMS] = {1, 1, 1, IN, HIDDEN};
    size_t b1_dims[TENSOR_DIMS] = {1, 1, 1, 1,  HIDDEN};
    size_t w2_dims[TENSOR_DIMS] = {1, 1, 1, HIDDEN, OUT};
    size_t b2_dims[TENSOR_DIMS] = {1, 1, 1, 1,  OUT};
    add_input_to_dag(&dag, W1, create_tensor(w1_dims));
    add_input_to_dag(&dag, b1, create_tensor(b1_dims));
    add_input_to_dag(&dag, W2, create_tensor(w2_dims));
    add_input_to_dag(&dag, b2, create_tensor(b2_dims));

    initialize_parameters(&dag);                     /* variance from topology + orthogonal matmul weights */
    emergence_promoting_initialization(&dag, 1.1f);  /* optional rescaling */

    /* --- training loop --- */
    for (int step = 0; step < 1000; step++) {
        add_input_to_dag(&dag, x, x_batch);          /* {1,1,1,BATCH,IN}  */
        add_input_to_dag(&dag, target, y_batch);     /* {1,1,1,BATCH,OUT} */

        forward(&dag);
        backward(&dag);
        optimize_weights(&dag);
        clear_memory(&dag);   /* frees the step's tape, keeps the parameters */
    }

    return 0;
}
```

The pattern is: build the graph once, attach and initialize parameters once, then each step re-feed the inputs, run `forward` / `backward` / `optimize_weights`, and `clear_memory` to release that step's tape while the parameter tensors persist. Swapping in other operators is the same recipe — add the node, wire its edges in argument order, and the forward/backward passes pick it up automatically.

## Core API

**Graph construction**

- `DirectedAcyclicGraph create_directed_acyclic_graph(void)` — returns an empty graph.
- `size_t add_node(dag, op_type, config)` — adds a node and returns its index; pass `NULL` for `config` on ops that take none.
- `void add_edge(dag, src_node, dst_node)` — connects an output to an input. **Edge order is the argument order**: for matmul the first edge is the left operand, for zoneout the first edge is the current state and the second is the previous state.
- `void add_input_to_dag(dag, node_index, tensor)` — attaches a tensor to an `INPUT` or `PARAMETERS` node (it stores a copy). Call once per parameter to give it shape and storage; call each step to feed inputs and targets.

**Initialization**

- `void initialize_parameters(dag)` — initializes each parameter from the graph structure (see below).
- `void emergence_promoting_initialization(dag, alpha)` — optional depth rescaling of weight layers, based on the paper *Advancing Neural Network Performance through Emergence-Promoting Initialization Scheme*.

**Execution**

- `void forward(dag)` — runs the forward pass for the next time step.
- `void backward(dag)` — accumulates gradients in reverse.
- `void optimize_weights(dag)` — applies L1/L2 then the optimizer.  Set to 0 in the config to skip.
- `void clear_memory(dag)` — frees the tape and re-seeds parameter tape entries for the next step.

**Tensors**

- `create_tensor(dims)`, `copy_tensor(t)`, `free_tensor(&t)`
- `set_tensor_data_to_zero(&t)`, `set_tensor_data_to_ones(&t)`, `add_gaussian_noise_to_tensor(&t, mean, std)`
- `get_data_index(t, indices)` — flat offset for a multi-index.

## Operations

| Category | Operation types |
| --- | --- |
| Structural | `RESIZE`, `CONCATENATE`, `ADD`, `SCALAR_MULTIPLICATION`, `ELEMENTWISE_MULTIPLICATION`, `MATRIX_MULTIPLICATION`, `CONVOLUTION` |
| Activations | `SIGMOID`, `GELU`, `SWISH`, `MISH`, `GELU_SINE`, `GELU_SINC`, `MISH_SINE`, `MISH_SINC` |
| Normalization | `SOFTMAX`, `LAYER_NORM`, `RMS_LAYER_NORM` |
| Losses | `CATEGORICAL_CROSS_ENTROPY_LOSS`, `MEAN_SQUARE_ERROR`, `KL_DIVERGENCE` |
| Stochastic regularizers | `DROPOUT`, `DROPPATH`, `ZONEOUT` |
| Special | `INPUT`, `PARAMETERS` |

`ADD`, `ELEMENTWISE_MULTIPLICATION`, and `MATRIX_MULTIPLICATION` broadcast dimensions of size `1`. For matmul the last two dimensions are the matrix; leading dimensions are batched. Convolution is channel-first: dimension `0` is channels and the remaining dimensions are spatial; a kernel packs its `(out_channels * in_channels)` filters along dimension `0`.

## Optimizers

Set `parameters.optimizer.type` on each `PARAMETERS` node:

- **`OPTIMIZER_TYPE_STOCHASTIC_GRADIENT_DESCENT`** — plain SGD with a learning rate.
- **`OPTIMIZER_TYPE_ADAM`** — Adam with bias correction.
- **`OPTIMIZER_TYPE_MUON`** — momentum with Newton–Schulz orthogonalization of the update. Supports Nesterov momentum and a configurable number of iteration steps.
- **`OPTIMIZER_TYPE_ANO`** — sign-based adaptive update with decoupled weight decay.

Each parameter owns its optimizer state, so different layers can use different optimizers in the same graph.

## Regularization

**Weight penalties** are applied inside `optimize_weights` before the optimizer step, driven by two independent floats on each `PARAMETERS` node:

- `l2_strength` adds `l2_strength * w` to the gradient (ridge).
- `l1_strength` adds `l1_strength * sign(w)` to the gradient (lasso).

Both can be nonzero simultaneously and a strength of `0` skips that penalty entirely.

**Stochastic regularizers** are configured with a `StochasticRegularizationConfiguration` (`probability`, `sample_dimension`, `base_seed`):

- **Dropout** — per element inverted dropout: a fraction `probability` of activations are zeroed and the survivors are scaled by `1 / (1 - probability)`.
- **DropPath** — per-sample stochastic depth: whole samples along `sample_dimension` are dropped or kept as a unit, scaled the same way. Use this on a residual branch and add the identity path with `ADD`.
- **Zoneout** — for RNNs: with probability `probability` a unit keeps its previous state instead of the newly computed one, passing that state and its gradient straight through the timestep. It takes two inputs — add the current-state edge first and the previous-state edge second.

These three derive their per-call random seed from `base_seed` combined with the operation's time step and node index, so the mask changes every step and differs across layers while remaining identical between a forward pass and its backward pass.

There is no separate inference mode. To evaluate without stochasticity, set `probability = 0` (Dropout and DropPath become the identity; Zoneout always takes the current state) or build an evaluation graph without these nodes.

## Initialization

`initialize_parameters` inspects what each parameter feeds into, up to three operations downstream, and chooses a strategy based on these rules:

- Parameters used only in `ADD` (biases) are left at zero.
- A relu-like activation (GELU/Swish/Mish) downstream selects He-style variance `2 / fan_in`.
- A softmax or cross-entropy downstream selects `1 / fan_in`.
- Otherwise it uses Xavier/Glorot variance `2 / (fan_in + fan_out)`.
- Convolution weights compute fan-in/fan-out from channel and kernel extents.
- Matmul weights with at least as many rows as columns are additionally made column-orthonormal via Householder QR.

`emergence_promoting_initialization(dag, alpha)` is an optional pass that ranks weight layers by depth and scales them geometrically around the middle layer by powers of `alpha`.

## Notes and limitations

- `TENSOR_DIMS` defaults to `5`. Prefer `RESIZE` over lowering it. Only change the constant directly if you are sure nothing in the project needs five dimensions.
- `MEAN_SQUARE_ERROR` reduces over the entire tensor, but `CATEGORICAL_CROSS_ENTROPY_LOSS` and `KL_DIVERGENCE` score a single distribution along the last dimension. Batch them by accumulating gradients over several forward/backward passes before calling `optimize_weights` and `clear_memory`.
- Normalization `gamma`/`beta` are ordinary parameters, and the structure-aware initializer fills them like weights rather than at one/zero, so set them yourself after `initialize_parameters` if you want the conventional unit-scale start.
- Computation is `float`, but reductions in normalization and softmax accumulate in `double` internally for stability.
- The engine is single-threaded and CPU-only, written for clarity and portability rather than peak throughput.
- There is no inference/training mode yet.
