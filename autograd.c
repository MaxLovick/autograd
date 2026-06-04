#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

float pcg_uniform_number_generator(float minimum, float maximum) {
    static uint64_t pcg_state = 0x853c49e6748fea9bULL;
    static uint64_t pcg_inc = 0xda3e39cb94b95bdbULL;

    uint64_t old_pcg_state = pcg_state;
    pcg_state = old_pcg_state * 6364136223846793005ULL + (pcg_inc | 1);
    uint32_t xorshifted = ((old_pcg_state >> 18u) ^ old_pcg_state) >> 27u;
    uint32_t rotation = old_pcg_state >> 59u;
    uint32_t bits = (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31));

    uint32_t mantissa_bits = bits >> 8;
    float random_number_between_0_and_1 = ldexpf((float)mantissa_bits, -24);

    return minimum + (maximum - minimum) * random_number_between_0_and_1;
}

float pcg_gaussian_number_generator(float mean, float std_dev) {
    float PI = 3.14159265358979323846f;

    float u1 = pcg_uniform_number_generator(0.0f, 1.0f);
    float u2 = pcg_uniform_number_generator(0.0f, 1.0f);
    if (u1 <= 0.0f) u1 = 1e-10f;
    float standard_normal = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)PI * u2);
    return mean + std_dev * standard_normal;
}

#define TENSOR_DIMS 4

typedef struct {
    size_t total_elements;
    size_t dims[TENSOR_DIMS];
    float* data;
} Tensor;

typedef enum {
    OPERATION_TYPE_INPUT,
    OPERATION_TYPE_PARAMETERS,
    OPERATION_TYPE_RESIZE,
    OPERATION_TYPE_CONCATENATE,
    OPERATION_TYPE_ADD,
    OPERATION_TYPE_SCALAR_MULTIPLICATION,
    OPERATION_TYPE_ELEMENTWISE_MULTIPLICATION,
    OPERATION_TYPE_MATRIX_MULTIPLICATION,
    OPERATION_TYPE_CONVOLUTION,
    OPERATION_TYPE_SIGMOID,
    OPERATION_TYPE_SOFTMAX,
    OPERATION_TYPE_GELU,
    OPERATION_TYPE_SWISH,
    OPERATION_TYPE_MISH,
    OPERATION_TYPE_GELU_SINE,
    OPERATION_TYPE_GELU_SINC,
    OPERATION_TYPE_MISH_SINE,
    OPERATION_TYPE_MISH_SINC,
    OPERATION_TYPE_LAYER_NORM,
    OPERATION_TYPE_RMS_LAYER_NORM,
    OPERATION_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS,
    OPERATION_TYPE_MEAN_SQUARE_ERROR,
    OPERATION_TYPE_KL_DIVERGENCE,
    OPERATION_TYPE_DROPOUT,
    OPERATION_TYPE_DROPPATH,
    OPERATION_TYPE_ZONEOUT,
} OperationType;

typedef enum {
    OPTIMIZER_TYPE_NONE,
    OPTIMIZER_TYPE_STOCHASTIC_GRADIENT_DESCENT,
    OPTIMIZER_TYPE_ADAM,
    OPTIMIZER_TYPE_MUON,
    OPTIMIZER_TYPE_ANO
} OptimizerType;

typedef struct {
    float learning_rate;
} StochasticGradientDescentConfiguration;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    uint64_t time_step;
    Tensor first_moment;
    Tensor second_moment;
} AdamConfiguration;

typedef struct {
    float learning_rate;
    float momentum;
    size_t newton_schulz_steps;
    bool nesterov;
    Tensor momentum_buffer;
} MuonConfiguration;

typedef struct {
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    Tensor first_moment;
    Tensor second_moment;
} AnoConfiguration;

typedef struct {
    float probability;
    size_t sample_dimension;
    uint64_t base_seed;
} StochasticRegularizationConfiguration;

typedef struct {
    OptimizerType type;
    union {
        StochasticGradientDescentConfiguration sgd;
        AdamConfiguration adam;
        MuonConfiguration muon;
        AnoConfiguration ano;
    } config;
} OptimizerConfiguration;

typedef struct {
    bool allow_parameter_updates;
    OptimizerConfiguration optimizer;
    float l1_strength;
    float l2_strength;
} ParameterConfiguration;

typedef struct {
    size_t new_dims[TENSOR_DIMS];
} ResizeConfiguration;

typedef struct {
    size_t concatenation_dimension;
} ConcatenationConfiguration;

typedef struct {
    size_t padding;
    size_t total_kernels;
} ConvolutionConfiguration;

typedef struct {
    size_t batch_dimension;
} SoftmaxConfiguration;

typedef struct {
    float epsilon;
    size_t normalization_dimension;
} LayerNormConfiguration;

typedef union {
    ParameterConfiguration parameters;
    ResizeConfiguration resize;
    ConcatenationConfiguration concatenation;
    ConvolutionConfiguration convolution;
    SoftmaxConfiguration softmax;
    LayerNormConfiguration layer_norm;
    StochasticRegularizationConfiguration stochastic_regularization;
} OperationConfiguration;

typedef struct {
    OperationType op_type;
    OperationConfiguration op_config;
    size_t total_src_nodes;
    size_t total_dst_nodes;
    size_t* src_node_indices;
    size_t* dst_node_indices;
    size_t total_tape_entries;
    size_t* tape_entry_indices;
} Node;

typedef struct {
    uint64_t time_step;
    size_t node_index;
    size_t total_src_tape_entries;
    size_t total_dst_tape_entries;
    size_t* src_tape_entry_indices;
    size_t* dst_tape_entry_indices;
    Tensor output_tensor;
    Tensor gradient_tensor;
} TapeEntry;

typedef struct {
    uint64_t current_time_step;
    size_t total_nodes;
    size_t total_tape_entries;
    Node* nodes;
    TapeEntry* tape_entries;
} DirectedAcyclicGraph;

Tensor create_tensor(const size_t dims[TENSOR_DIMS]) {
    Tensor tensor;
    if(dims) {
        tensor.total_elements = 1;
        for(size_t i = 0; i < TENSOR_DIMS; i++) {
            tensor.dims[i] = dims[i];
            tensor.total_elements *= dims[i];
        }
        if(tensor.total_elements > 0) {
            tensor.data = malloc(tensor.total_elements * sizeof(float));
        }
        else {
            tensor.data = NULL;
        }
    }
    else {
        tensor.total_elements = 0;
        for(size_t i = 0; i < TENSOR_DIMS; i++) {
            tensor.dims[i] = 0;
        }
        tensor.data = NULL;
    }
    return tensor;
}

Tensor copy_tensor(const Tensor tensor) {
    Tensor new_tensor = create_tensor(tensor.dims);
    for(size_t i = 0; i < new_tensor.total_elements; i++) {
        new_tensor.data[i] = tensor.data[i];
    }
    return new_tensor;
}

void free_tensor(Tensor* tensor) {
    free(tensor->data);
    tensor->data = NULL;

    tensor->total_elements = 0;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        tensor->dims[i] = 0;
    }
}

void set_tensor_data_to_zero(Tensor* tensor) {
    for(size_t i = 0; i < tensor->total_elements; i++) {
        tensor->data[i] = 0.0f;
    }
}

void set_tensor_data_to_ones(Tensor* tensor) {
    for(size_t i = 0; i < tensor->total_elements; i++) {
        tensor->data[i] = 1.0f;
    }
}

void add_gaussian_noise_to_tensor(Tensor* tensor, double mean, double std_dev) {
    for(size_t i = 0; i < tensor->total_elements; i++) {
        tensor->data[i] += pcg_gaussian_number_generator(mean, std_dev);
    }
}

size_t get_data_index(const Tensor tensor, const size_t indices[TENSOR_DIMS]) {
    size_t index = 0;
    size_t stride = 1;
    for (size_t i = TENSOR_DIMS; i-- > 0; ) {
        index += indices[i] * stride;
        stride *= tensor.dims[i];
    }
    return index;
}

bool make_columns_orthonormal_householder(Tensor* tensor, float epsilon) {
    const size_t ROW_DIMENSION = TENSOR_DIMS - 2;
    const size_t COLUMN_DIMENSION = TENSOR_DIMS - 1;
    size_t total_rows = tensor->dims[ROW_DIMENSION];
    size_t total_columns = tensor->dims[COLUMN_DIMENSION];
    if(total_rows < total_columns) { return false; }
    size_t total_slices = 1;
    for(size_t i = 0; i < ROW_DIMENSION; i++) {
        total_slices *= tensor->dims[i];
    }
    float* householder_vectors = malloc(total_rows * total_columns * sizeof(float));
    if(householder_vectors == NULL) { return false; }
    size_t indices[TENSOR_DIMS];
    for(size_t slice_index = 0; slice_index < total_slices; slice_index++) {
        size_t remainder = slice_index;
        for(size_t i = ROW_DIMENSION; i-- > 0;) {
            indices[i] = remainder % tensor->dims[i];
            remainder /= tensor->dims[i];
        }
        for(size_t column_index = 0; column_index < total_columns; column_index++) {
            indices[COLUMN_DIMENSION] = column_index;
            float column_squared_norm = 0.0f;
            for(size_t row_index = column_index; row_index < total_rows; row_index++) {
                indices[ROW_DIMENSION] = row_index;
                size_t element_offset = get_data_index(*tensor, indices);
                column_squared_norm += tensor->data[element_offset] * tensor->data[element_offset];
            }
            float column_norm = sqrtf(column_squared_norm);
            if(column_norm < epsilon) {
                free(householder_vectors);
                return false;
            }
            indices[ROW_DIMENSION] = column_index;
            size_t diagonal_element_offset = get_data_index(*tensor, indices);
            float diagonal_value = tensor->data[diagonal_element_offset];
            float sign = (diagonal_value >= 0.0f) ? 1.0f : -1.0f;
            float alpha = -sign * column_norm;
            householder_vectors[column_index * total_rows + column_index] = diagonal_value - alpha;
            for(size_t row_index = column_index + 1; row_index < total_rows; row_index++) {
                indices[ROW_DIMENSION] = row_index;
                size_t element_offset = get_data_index(*tensor, indices);
                householder_vectors[column_index * total_rows + row_index] = tensor->data[element_offset];
            }
            float v_squared_norm = 0.0f;
            for(size_t row_index = column_index; row_index < total_rows; row_index++) {
                float v_element = householder_vectors[column_index * total_rows + row_index];
                v_squared_norm += v_element * v_element;
            }
            float inv_v_norm = 1.0f / sqrtf(v_squared_norm);
            for(size_t row_index = column_index; row_index < total_rows; row_index++) {
                householder_vectors[column_index * total_rows + row_index] *= inv_v_norm;
            }
            for(size_t j = column_index; j < total_columns; j++) {
                indices[COLUMN_DIMENSION] = j;
                float dot = 0.0f;
                for(size_t row_index = column_index; row_index < total_rows; row_index++) {
                    indices[ROW_DIMENSION] = row_index;
                    size_t element_offset = get_data_index(*tensor, indices);
                    dot += householder_vectors[column_index * total_rows + row_index] * tensor->data[element_offset];
                }
                float two_times_dot = 2.0f * dot;
                for(size_t row_index = column_index; row_index < total_rows; row_index++) {
                    indices[ROW_DIMENSION] = row_index;
                    size_t element_offset = get_data_index(*tensor, indices);
                    float v_element = householder_vectors[column_index * total_rows + row_index];
                    tensor->data[element_offset] -= two_times_dot * v_element;
                }
            }
        }
        for(size_t column_index = 0; column_index < total_columns; column_index++) {
            indices[COLUMN_DIMENSION] = column_index;
            for(size_t row_index = 0; row_index < total_rows; row_index++) {
                indices[ROW_DIMENSION] = row_index;
                size_t element_offset = get_data_index(*tensor, indices);
                tensor->data[element_offset] = (row_index == column_index) ? 1.0f : 0.0f;
            }
        }
        for(size_t k = total_columns; k-- > 0;) {
            for(size_t j = k; j < total_columns; j++) {
                indices[COLUMN_DIMENSION] = j;
                float dot = 0.0f;
                for(size_t row_index = k; row_index < total_rows; row_index++) {
                    indices[ROW_DIMENSION] = row_index;
                    size_t element_offset = get_data_index(*tensor, indices);
                    dot += householder_vectors[k * total_rows + row_index] * tensor->data[element_offset];
                }
                float two_times_dot = 2.0f * dot;
                for(size_t row_index = k; row_index < total_rows; row_index++) {
                    indices[ROW_DIMENSION] = row_index;
                    size_t element_offset = get_data_index(*tensor, indices);
                    float v_element = householder_vectors[k * total_rows + row_index];
                    tensor->data[element_offset] -= two_times_dot * v_element;
                }
            }
        }
    }
    free(householder_vectors);
    return true;
}

Tensor resize(const Tensor input, const size_t new_dims[TENSOR_DIMS]) {
    Tensor output = create_tensor(new_dims);
    for(size_t i = 0; i < input.total_elements; i++) {
        output.data[i] = input.data[i];
    }
    return output;
}

void resize_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        input_gradients->data[i] += upstream_gradients.data[i];
    }
}

Tensor concatenate(const Tensor input1, const Tensor input2, const size_t concatenation_dimension) {
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        output_dims[i] = (i == concatenation_dimension) ? (input1.dims[i] + input2.dims[i]) : input1.dims[i];
    }
    Tensor output = create_tensor(output_dims);

    size_t total_inner_blocks = 1;
    for(size_t i = 0; i < concatenation_dimension; i++) {
        total_inner_blocks *= input1.dims[i];
    }
    size_t inner_block_size = 1;
    for(size_t i = (concatenation_dimension + 1); i < TENSOR_DIMS; i++) {
        inner_block_size *= input1.dims[i];
    }
    size_t input1_inner_block_size = input1.dims[concatenation_dimension] * inner_block_size;
    size_t input2_inner_block_size = input2.dims[concatenation_dimension] * inner_block_size;
    size_t output_inner_block_size = output.dims[concatenation_dimension] * inner_block_size;

    for(size_t inner_block = 0; inner_block < total_inner_blocks; inner_block++) {
        for(size_t i = 0; i < input1_inner_block_size; i++) {
            output.data[(inner_block * output_inner_block_size) + i] = input1.data[(inner_block * input1_inner_block_size) + i];
        }
        for(size_t i = 0; i < input2_inner_block_size; i++) {
            output.data[(inner_block * output_inner_block_size) + input1_inner_block_size + i] = input2.data[(inner_block * input2_inner_block_size) + i];
        }
    }
    return output;
}

void concatenate_gradients(const Tensor input1, const Tensor input2, const Tensor upstream_gradients, Tensor* input1_gradients, Tensor* input2_gradients, const size_t concatenation_dimension) {
    size_t total_inner_blocks = 1;
    for(size_t i = 0; i < concatenation_dimension; i++) {
        total_inner_blocks *= input1.dims[i];
    }
    size_t inner_block_size = 1;
    for(size_t i = (concatenation_dimension + 1); i < TENSOR_DIMS; i++) {
        inner_block_size *= input1.dims[i];
    }
    size_t input1_inner_block_size = input1.dims[concatenation_dimension] * inner_block_size;
    size_t input2_inner_block_size = input2.dims[concatenation_dimension] * inner_block_size;
    size_t upstream_gradients_inner_block_size = upstream_gradients.dims[concatenation_dimension] * inner_block_size;

    for(size_t inner_block = 0; inner_block < total_inner_blocks; inner_block++) {
        for(size_t i = 0; i < input1_inner_block_size; i++) {
            input1_gradients->data[(inner_block * input1_inner_block_size) + i] += upstream_gradients.data[(inner_block * upstream_gradients_inner_block_size) + i];
        }
        for(size_t i = 0; i < input2_inner_block_size; i++) {
            input2_gradients->data[(inner_block * input2_inner_block_size) + i] += upstream_gradients.data[(inner_block * upstream_gradients_inner_block_size) + input1_inner_block_size + i];
        }
    }
}

Tensor add(const Tensor input1, const Tensor input2) {
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(input1.dims[i] == input2.dims[i]) { output_dims[i] = input1.dims[i]; }
        else if(input1.dims[i] > 1 && input2.dims[i] == 1) { output_dims[i] = input1.dims[i]; }
        else if(input1.dims[i] == 1 && input2.dims[i] > 1) { output_dims[i] = input2.dims[i]; }
    }
    Tensor output = create_tensor(output_dims);

    size_t input1_indices[TENSOR_DIMS];
    size_t input2_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_indices[i] = 0; }

    for(size_t output_index = 0; output_index < output.total_elements; output_index++) {
        for(size_t i = 0; i < TENSOR_DIMS; i++) {
            input1_indices[i] = (input1.dims[i] == 1) ? 0 : output_indices[i];
            input2_indices[i] = (input2.dims[i] == 1) ? 0 : output_indices[i];
        }

        size_t input1_index = get_data_index(input1, input1_indices);
        size_t input2_index = get_data_index(input2, input2_indices);

        output.data[output_index] = input1.data[input1_index] + input2.data[input2_index];

        for(size_t i = TENSOR_DIMS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] < output.dims[i]) { break; }
            output_indices[i] = 0;
        }
    }
    return output;
}

void add_gradients(const Tensor input1, const Tensor input2, const Tensor upstream_gradients, Tensor* input1_gradients, Tensor* input2_gradients) {
    size_t input1_indices[TENSOR_DIMS];
    size_t input2_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_indices[i] = 0; }

    for(size_t output_index = 0; output_index < upstream_gradients.total_elements; output_index++) {
        for(size_t i = 0; i < TENSOR_DIMS; i++) {
            input1_indices[i] = (input1.dims[i] == 1) ? 0 : output_indices[i];
            input2_indices[i] = (input2.dims[i] == 1) ? 0 : output_indices[i];
        }

        size_t input1_index = get_data_index(input1, input1_indices);
        size_t input2_index = get_data_index(input2, input2_indices);

        input1_gradients->data[input1_index] += upstream_gradients.data[output_index];
        input2_gradients->data[input2_index] += upstream_gradients.data[output_index];

        for(size_t i = TENSOR_DIMS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] < upstream_gradients.dims[i]) { break; }
            output_indices[i] = 0;
        }
    }
}

Tensor scalar_multiplication(const Tensor scalar_input, const Tensor tensor_input) {
    Tensor output = create_tensor(tensor_input.dims);
    for(size_t i = 0; i < tensor_input.total_elements; i++) {
        output.data[i] = tensor_input.data[i] * scalar_input.data[0];
    }
    return output;
}

void scalar_multiplication_gradients(const Tensor scalar_input, const Tensor tensor_input, const Tensor upstream_gradients, Tensor* scalar_input_gradients, Tensor* tensor_input_gradients) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        scalar_input_gradients->data[0] += tensor_input.data[i] * upstream_gradients.data[i];
        tensor_input_gradients->data[i] += scalar_input.data[0] * upstream_gradients.data[i];
    }
}

Tensor elementwise_mul(const Tensor input1, const Tensor input2) {
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(input1.dims[i] == input2.dims[i]) { output_dims[i] = input1.dims[i]; }
        else if(input1.dims[i] > 1 && input2.dims[i] == 1) { output_dims[i] = input1.dims[i]; }
        else if(input1.dims[i] == 1 && input2.dims[i] > 1) { output_dims[i] = input2.dims[i]; }
    }
    Tensor output = create_tensor(output_dims);

    size_t input1_indices[TENSOR_DIMS];
    size_t input2_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_indices[i] = 0; }

    for(size_t output_index = 0; output_index < output.total_elements; output_index++) {
        for(size_t i = 0; i < TENSOR_DIMS; i++) {
            input1_indices[i] = (input1.dims[i] == 1) ? 0 : output_indices[i];
            input2_indices[i] = (input2.dims[i] == 1) ? 0 : output_indices[i];
        }

        size_t input1_index = get_data_index(input1, input1_indices);
        size_t input2_index = get_data_index(input2, input2_indices);

        output.data[output_index] = input1.data[input1_index] * input2.data[input2_index];

        for(size_t i = TENSOR_DIMS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] < output.dims[i]) { break; }
            output_indices[i] = 0;
        }
    }
    return output;
}

void elementwise_mul_gradients(const Tensor input1, const Tensor input2, const Tensor upstream_gradients, Tensor* input1_gradients, Tensor* input2_gradients) {
    size_t input1_indices[TENSOR_DIMS];
    size_t input2_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_indices[i] = 0; }

    for(size_t output_index = 0; output_index < upstream_gradients.total_elements; output_index++) {
        for(size_t i = 0; i < TENSOR_DIMS; i++) {
            input1_indices[i] = (input1.dims[i] == 1) ? 0 : output_indices[i];
            input2_indices[i] = (input2.dims[i] == 1) ? 0 : output_indices[i];
        }

        size_t input1_index = get_data_index(input1, input1_indices);
        size_t input2_index = get_data_index(input2, input2_indices);

        input1_gradients->data[input1_index] += input2.data[input2_index] * upstream_gradients.data[output_index];
        input2_gradients->data[input2_index] += input1.data[input1_index] * upstream_gradients.data[output_index];

        for(size_t i = TENSOR_DIMS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] < upstream_gradients.dims[i]) { break; }
            output_indices[i] = 0;
        }
    }
}

Tensor matrix_multiplication(const Tensor input1, const Tensor input2) {
    const size_t ROW_DIMENSION = TENSOR_DIMS - 2;
    const size_t COLUMN_DIMENSION = TENSOR_DIMS - 1;
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < ROW_DIMENSION; i++) {
        if(input1.dims[i] == input2.dims[i]) { output_dims[i] = input1.dims[i]; }
        else if(input1.dims[i] > 1 && input2.dims[i] == 1) { output_dims[i] = input1.dims[i]; }
        else if(input1.dims[i] == 1 && input2.dims[i] > 1) { output_dims[i] = input2.dims[i]; }
    }
    output_dims[ROW_DIMENSION] = input1.dims[ROW_DIMENSION];
    output_dims[COLUMN_DIMENSION] = input2.dims[COLUMN_DIMENSION];
    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zero(&output);

    size_t shared_dim = input1.dims[COLUMN_DIMENSION];
    size_t input1_indices[TENSOR_DIMS];
    size_t input2_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_indices[i] = 0; }

    for(size_t output_index = 0; output_index < output.total_elements; output_index++) {
        for(size_t i = 0; i < ROW_DIMENSION; i++) {
            input1_indices[i] = (input1.dims[i] == 1) ? 0 : output_indices[i];
            input2_indices[i] = (input2.dims[i] == 1) ? 0 : output_indices[i];
        }
        input1_indices[ROW_DIMENSION] = output_indices[ROW_DIMENSION];
        input2_indices[COLUMN_DIMENSION] = output_indices[COLUMN_DIMENSION];

        for(size_t k = 0; k < shared_dim; k++) {
            input1_indices[COLUMN_DIMENSION] = k;
            input2_indices[ROW_DIMENSION] = k;
            size_t input1_offset = get_data_index(input1, input1_indices);
            size_t input2_offset = get_data_index(input2, input2_indices);
            output.data[output_index] += input1.data[input1_offset] * input2.data[input2_offset];
        }

        for(size_t i = TENSOR_DIMS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] < output.dims[i]) { break; }
            output_indices[i] = 0;
        }
    }
    return output;
}

void matrix_multiplication_gradients(const Tensor input1, const Tensor input2, const Tensor upstream_gradients, Tensor* input1_gradients, Tensor* input2_gradients) {
    const size_t ROW_DIMENSION = TENSOR_DIMS - 2;
    const size_t COLUMN_DIMENSION = TENSOR_DIMS - 1;
    size_t shared_dim = input1.dims[COLUMN_DIMENSION];
    size_t input1_indices[TENSOR_DIMS];
    size_t input2_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_indices[i] = 0; }
    for(size_t output_index = 0; output_index < upstream_gradients.total_elements; output_index++) {
        for(size_t i = 0; i < ROW_DIMENSION; i++) {
            input1_indices[i] = (input1.dims[i] == 1) ? 0 : output_indices[i];
            input2_indices[i] = (input2.dims[i] == 1) ? 0 : output_indices[i];
        }
        input1_indices[ROW_DIMENSION] = output_indices[ROW_DIMENSION];
        input2_indices[COLUMN_DIMENSION] = output_indices[COLUMN_DIMENSION];
        for(size_t k = 0; k < shared_dim; k++) {
            input1_indices[COLUMN_DIMENSION] = k;
            input2_indices[ROW_DIMENSION] = k;
            size_t input1_offset = get_data_index(input1, input1_indices);
            size_t input2_offset = get_data_index(input2, input2_indices);
            input1_gradients->data[input1_offset] += upstream_gradients.data[output_index] * input2.data[input2_offset];
            input2_gradients->data[input2_offset] += upstream_gradients.data[output_index] * input1.data[input1_offset];
        }
        for(size_t i = TENSOR_DIMS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] < upstream_gradients.dims[i]) { break; }
            output_indices[i] = 0;
        }
    }
}

Tensor convolution(const Tensor tensor_input, const Tensor kernel_input, size_t padding, size_t total_kernels) {
    const size_t CHANNEL_DIMENSION = 0;
    const size_t SPATIAL_START_INDEX = 1;
    size_t num_spatial_dims = TENSOR_DIMS - SPATIAL_START_INDEX;

    size_t in_channels = tensor_input.dims[CHANNEL_DIMENSION];

    size_t output_dims[TENSOR_DIMS];
    output_dims[CHANNEL_DIMENSION] = total_kernels;
    for(size_t i = 0; i < num_spatial_dims; i++) {
        size_t input_extent = tensor_input.dims[SPATIAL_START_INDEX + i];
        size_t kernel_extent = kernel_input.dims[SPATIAL_START_INDEX + i];
        size_t padded_input = input_extent + (2 * padding);
        output_dims[SPATIAL_START_INDEX + i] = (padded_input >= kernel_extent) ? ((padded_input - kernel_extent) + 1) : 0;
    }

    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zero(&output);

    size_t input_indices[TENSOR_DIMS];
    size_t kernel_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    size_t output_position[TENSOR_DIMS];
    size_t kernel_position[TENSOR_DIMS];

    for(size_t kernel_number = 0; kernel_number < total_kernels; kernel_number++) {
        for(size_t i = 0; i < num_spatial_dims; i++) { output_position[i] = 0; }

        bool has_more_output = true;
        while(has_more_output) {
            double accumulator = 0.0;

            for(size_t channel = 0; channel < in_channels; channel++) {
                for(size_t i = 0; i < num_spatial_dims; i++) { kernel_position[i] = 0; }

                bool has_more_kernel = true;
                while(has_more_kernel) {
                    bool inside_input = true;
                    for(size_t i = 0; i < num_spatial_dims; i++) {
                        size_t padded_coordinate = output_position[i] + kernel_position[i];
                        if(padded_coordinate < padding) { inside_input = false; break; }
                        size_t coordinate = padded_coordinate - padding;
                        if(coordinate >= tensor_input.dims[SPATIAL_START_INDEX + i]) { inside_input = false; break; }
                        input_indices[SPATIAL_START_INDEX + i] = coordinate;
                    }

                    if(inside_input) {
                        input_indices[CHANNEL_DIMENSION] = channel;
                        kernel_indices[CHANNEL_DIMENSION] = (kernel_number * in_channels) + channel;
                        for(size_t i = 0; i < num_spatial_dims; i++) {
                            kernel_indices[SPATIAL_START_INDEX + i] = kernel_position[i];
                        }
                        size_t input_offset = get_data_index(tensor_input, input_indices);
                        size_t kernel_offset = get_data_index(kernel_input, kernel_indices);
                        accumulator += (double)tensor_input.data[input_offset] * (double)kernel_input.data[kernel_offset];
                    }

                    if(num_spatial_dims == 0) { has_more_kernel = false; break; }
                    for(size_t i = num_spatial_dims; i-- > 0; ) {
                        kernel_position[i]++;
                        if(kernel_position[i] < kernel_input.dims[SPATIAL_START_INDEX + i]) { break; }
                        kernel_position[i] = 0;
                        if(i == 0) { has_more_kernel = false; }
                    }
                }
            }

            output_indices[CHANNEL_DIMENSION] = kernel_number;
            for(size_t i = 0; i < num_spatial_dims; i++) { output_indices[SPATIAL_START_INDEX + i] = output_position[i]; }
            size_t output_offset = get_data_index(output, output_indices);
            output.data[output_offset] = (float)accumulator;

            if(num_spatial_dims == 0) { has_more_output = false; break; }
            for(size_t i = num_spatial_dims; i-- > 0; ) {
                output_position[i]++;
                if(output_position[i] < output_dims[SPATIAL_START_INDEX + i]) { break; }
                output_position[i] = 0;
                if(i == 0) { has_more_output = false; }
            }
        }
    }
    return output;
}

void convolution_gradients(const Tensor tensor_input, const Tensor kernel_input, const Tensor upstream_gradients, Tensor* tensor_input_gradients, Tensor* kernel_input_gradients, size_t padding, size_t total_kernels) {
    const size_t CHANNEL_DIMENSION = 0;
    const size_t SPATIAL_START_INDEX = 1;
    size_t num_spatial_dims = TENSOR_DIMS - SPATIAL_START_INDEX;

    size_t in_channels = tensor_input.dims[CHANNEL_DIMENSION];

    size_t input_indices[TENSOR_DIMS];
    size_t kernel_indices[TENSOR_DIMS];
    size_t output_indices[TENSOR_DIMS];
    size_t output_position[TENSOR_DIMS];
    size_t kernel_position[TENSOR_DIMS];

    for(size_t kernel_number = 0; kernel_number < total_kernels; kernel_number++) {
        for(size_t i = 0; i < num_spatial_dims; i++) { output_position[i] = 0; }

        bool has_more_output = true;
        while(has_more_output) {
            output_indices[CHANNEL_DIMENSION] = kernel_number;
            for(size_t i = 0; i < num_spatial_dims; i++) { output_indices[SPATIAL_START_INDEX + i] = output_position[i]; }
            size_t output_offset = get_data_index(upstream_gradients, output_indices);
            double upstream_gradient = (double)upstream_gradients.data[output_offset];

            for(size_t channel = 0; channel < in_channels; channel++) {
                for(size_t i = 0; i < num_spatial_dims; i++) { kernel_position[i] = 0; }

                bool has_more_kernel = true;
                while(has_more_kernel) {
                    bool inside_input = true;
                    for(size_t i = 0; i < num_spatial_dims; i++) {
                        size_t padded_coordinate = output_position[i] + kernel_position[i];
                        if(padded_coordinate < padding) { inside_input = false; break; }
                        size_t coordinate = padded_coordinate - padding;
                        if(coordinate >= tensor_input.dims[SPATIAL_START_INDEX + i]) { inside_input = false; break; }
                        input_indices[SPATIAL_START_INDEX + i] = coordinate;
                    }

                    if(inside_input) {
                        input_indices[CHANNEL_DIMENSION] = channel;
                        kernel_indices[CHANNEL_DIMENSION] = (kernel_number * in_channels) + channel;
                        for(size_t i = 0; i < num_spatial_dims; i++) {
                            kernel_indices[SPATIAL_START_INDEX + i] = kernel_position[i];
                        }
                        size_t input_offset = get_data_index(tensor_input, input_indices);
                        size_t kernel_offset = get_data_index(kernel_input, kernel_indices);

                        tensor_input_gradients->data[input_offset] += (float)((double)kernel_input.data[kernel_offset] * upstream_gradient);
                        kernel_input_gradients->data[kernel_offset] += (float)((double)tensor_input.data[input_offset] * upstream_gradient);
                    }

                    if(num_spatial_dims == 0) { has_more_kernel = false; break; }
                    for(size_t i = num_spatial_dims; i-- > 0; ) {
                        kernel_position[i]++;
                        if(kernel_position[i] < kernel_input.dims[SPATIAL_START_INDEX + i]) { break; }
                        kernel_position[i] = 0;
                        if(i == 0) { has_more_kernel = false; }
                    }
                }
            }

            if(num_spatial_dims == 0) { has_more_output = false; break; }
            for(size_t i = num_spatial_dims; i-- > 0; ) {
                output_position[i]++;
                if(output_position[i] < upstream_gradients.dims[SPATIAL_START_INDEX + i]) { break; }
                output_position[i] = 0;
                if(i == 0) { has_more_output = false; }
            }
        }
    }
}

Tensor layer_norm(const Tensor input, const Tensor gamma, const Tensor beta, float epsilon, size_t norm_dim) {
    Tensor output = create_tensor(input.dims);

    size_t outer_count = 1;
    size_t inner_count = 1;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(i < norm_dim) { outer_count *= input.dims[i]; }
        else { inner_count *= input.dims[i]; }
    }

    for(size_t outer_count_index = 0; outer_count_index < outer_count; outer_count_index++) {
        size_t outer_offset = outer_count_index * inner_count;

        double value_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            value_sum += (double)input.data[outer_offset + i];
        }
        double mean = value_sum / (double)inner_count;

        double squared_difference_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double centered_value = (double)input.data[outer_offset + i] - mean;
            squared_difference_sum += centered_value * centered_value;
        }
        double variance = squared_difference_sum / (double)inner_count;
        double inverse_std_dev = 1.0 / sqrt(variance + (double)epsilon);

        for(size_t i = 0; i < inner_count; i++) {
            double centered_value = (double)input.data[outer_offset + i] - mean;
            double normalized_value = centered_value * inverse_std_dev;
            double scaled_value = (normalized_value * (double)gamma.data[i]) + (double)beta.data[i];
            output.data[outer_offset + i] = (float)scaled_value;
        }
    }

    return output;
}

void layer_norm_gradients(const Tensor input, const Tensor gamma, const Tensor beta, const Tensor upstream_gradients, Tensor* input_gradients, Tensor* gamma_gradients, Tensor* beta_gradients, float epsilon, size_t norm_dim) {
    size_t outer_count = 1;
    size_t inner_count = 1;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(i < norm_dim) { outer_count *= input.dims[i]; }
        else { inner_count *= input.dims[i]; }
    }

    for(size_t outer_count_index = 0; outer_count_index < outer_count; outer_count_index++) {
        size_t outer_offset = outer_count_index * inner_count;

        double value_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            value_sum += (double)input.data[outer_offset + i];
        }
        double mean = value_sum / (double)inner_count;

        double squared_difference_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double centered_value = (double)input.data[outer_offset + i] - mean;
            squared_difference_sum += centered_value * centered_value;
        }
        double variance = squared_difference_sum / (double)inner_count;
        double inverse_std_dev = 1.0 / sqrt(variance + (double)epsilon);

        double normalized_gradient_sum = 0.0;
        double normalized_gradient_dot_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double upstream_gradient = (double)upstream_gradients.data[outer_offset + i];
            double normalized_value = ((double)input.data[outer_offset + i] - mean) * inverse_std_dev;

            beta_gradients->data[i] += (float)upstream_gradient;
            gamma_gradients->data[i] += (float)(upstream_gradient * normalized_value);

            double normalized_gradient = upstream_gradient * (double)gamma.data[i];
            normalized_gradient_sum += normalized_gradient;
            normalized_gradient_dot_sum += normalized_gradient * normalized_value;
        }
        double mean_normalized_gradient = normalized_gradient_sum / (double)inner_count;
        double mean_normalized_gradient_dot = normalized_gradient_dot_sum / (double)inner_count;

        for(size_t i = 0; i < inner_count; i++) {
            double upstream_gradient = (double)upstream_gradients.data[outer_offset + i];
            double normalized_value = ((double)input.data[outer_offset + i] - mean) * inverse_std_dev;
            double normalized_gradient = upstream_gradient * (double)gamma.data[i];

            double input_gradient = inverse_std_dev * (normalized_gradient - mean_normalized_gradient - (normalized_value * mean_normalized_gradient_dot));
            input_gradients->data[outer_offset + i] += (float)input_gradient;
        }
    }
}

Tensor rms_norm(const Tensor input, const Tensor gamma, float epsilon, size_t norm_dim) {
    Tensor output = create_tensor(input.dims);

    size_t outer_count = 1;
    size_t inner_count = 1;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(i < norm_dim) { outer_count *= input.dims[i]; }
        else { inner_count *= input.dims[i]; }
    }

    for(size_t outer_count_index = 0; outer_count_index < outer_count; outer_count_index++) {
        size_t outer_offset = outer_count_index * inner_count;

        double squared_value_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double value = (double)input.data[outer_offset + i];
            squared_value_sum += value * value;
        }
        double mean_square = squared_value_sum / (double)inner_count;
        double inverse_rms = 1.0 / sqrt(mean_square + (double)epsilon);

        for(size_t i = 0; i < inner_count; i++) {
            double normalized_value = (double)input.data[outer_offset + i] * inverse_rms;
            double scaled_value = normalized_value * (double)gamma.data[i];
            output.data[outer_offset + i] = (float)scaled_value;
        }
    }

    return output;
}

void rms_norm_gradients(const Tensor input, const Tensor gamma, const Tensor upstream_gradients, Tensor* input_gradients, Tensor* gamma_gradients, float epsilon, size_t norm_dim) {
    size_t outer_count = 1;
    size_t inner_count = 1;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(i < norm_dim) { outer_count *= input.dims[i]; }
        else { inner_count *= input.dims[i]; }
    }

    for(size_t outer_count_index = 0; outer_count_index < outer_count; outer_count_index++) {
        size_t outer_offset = outer_count_index * inner_count;

        double squared_value_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double value = (double)input.data[outer_offset + i];
            squared_value_sum += value * value;
        }
        double mean_square = squared_value_sum / (double)inner_count;
        double inverse_rms = 1.0 / sqrt(mean_square + (double)epsilon);

        double normalized_gradient_dot_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double upstream_gradient = (double)upstream_gradients.data[outer_offset + i];
            double normalized_value = (double)input.data[outer_offset + i] * inverse_rms;

            gamma_gradients->data[i] += (float)(upstream_gradient * normalized_value);

            double normalized_gradient = upstream_gradient * (double)gamma.data[i];
            normalized_gradient_dot_sum += normalized_gradient * normalized_value;
        }
        double mean_normalized_gradient_dot = normalized_gradient_dot_sum / (double)inner_count;

        for(size_t i = 0; i < inner_count; i++) {
            double upstream_gradient = (double)upstream_gradients.data[outer_offset + i];
            double normalized_value = (double)input.data[outer_offset + i] * inverse_rms;
            double normalized_gradient = upstream_gradient * (double)gamma.data[i];

            double input_gradient = inverse_rms * (normalized_gradient - (normalized_value * mean_normalized_gradient_dot));
            input_gradients->data[outer_offset + i] += (float)input_gradient;
        }
    }
}

Tensor softmax(const Tensor input, size_t norm_dim) {
    Tensor output = create_tensor(input.dims);

    size_t outer_count = 1;
    size_t inner_count = 1;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(i < norm_dim) { outer_count *= input.dims[i]; }
        else { inner_count *= input.dims[i]; }
    }

    for(size_t outer_count_index = 0; outer_count_index < outer_count; outer_count_index++) {
        size_t outer_offset = outer_count_index * inner_count;

        float max_value = input.data[outer_offset];
        for(size_t i = 0; i < inner_count; i++) {
            if(input.data[outer_offset + i] > max_value) {
                max_value = input.data[outer_offset + i];
            }
        }

        double exp_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double exp_value = exp((double)input.data[outer_offset + i] - (double)max_value);
            output.data[outer_offset + i] = (float)exp_value;
            exp_sum += exp_value;
        }

        double inverse_exp_sum = 1.0 / exp_sum;
        for(size_t i = 0; i < inner_count; i++) {
            double normalized_value = (double)output.data[outer_offset + i] * inverse_exp_sum;
            output.data[outer_offset + i] = (float)normalized_value;
        }
    }

    return output;
}

void softmax_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients, size_t norm_dim) {
    size_t outer_count = 1;
    size_t inner_count = 1;
    for(size_t i = 0; i < TENSOR_DIMS; i++) {
        if(i < norm_dim) { outer_count *= input.dims[i]; }
        else { inner_count *= input.dims[i]; }
    }

    for(size_t outer_count_index = 0; outer_count_index < outer_count; outer_count_index++) {
        size_t outer_offset = outer_count_index * inner_count;

        float max_value = input.data[outer_offset];
        for(size_t i = 0; i < inner_count; i++) {
            if(input.data[outer_offset + i] > max_value) {
                max_value = input.data[outer_offset + i];
            }
        }

        double exp_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            exp_sum += exp((double)input.data[outer_offset + i] - (double)max_value);
        }
        double inverse_exp_sum = 1.0 / exp_sum;

        double gradient_dot_softmax_sum = 0.0;
        for(size_t i = 0; i < inner_count; i++) {
            double softmax_value = exp((double)input.data[outer_offset + i] - (double)max_value) * inverse_exp_sum;
            double upstream_gradient = (double)upstream_gradients.data[outer_offset + i];
            gradient_dot_softmax_sum += upstream_gradient * softmax_value;
        }

        for(size_t i = 0; i < inner_count; i++) {
            double softmax_value = exp((double)input.data[outer_offset + i] - (double)max_value) * inverse_exp_sum;
            double upstream_gradient = (double)upstream_gradients.data[outer_offset + i];
            double input_gradient = softmax_value * (upstream_gradient - gradient_dot_softmax_sum);
            input_gradients->data[outer_offset + i] += (float)input_gradient;
        }
    }
}

Tensor sigmoid(const Tensor input) {
    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        if(input.data[i] >= 0.0f) {
            output.data[i] = 1.0 / (1.0 + expf(-1.0 * x));
        }
        else {
            x = expf(x);
            output.data[i] = x / (1.0 + x);
        }
    }
    return output;
}

void sigmoid_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    for(size_t i = 0; i < input.total_elements; i++) {
        float x = input.data[i];
        float s = (x >= 0.0f) ? (1.0f / (1.0f + expf(-x))) : (expf(x) / (1.0f + expf(x)));
        input_gradients->data[i] += upstream_gradients.data[i] * s * (1.0f - s);
    }
}

Tensor gelu(const Tensor input) {
    Tensor output = create_tensor(input.dims);
    const float INV_SQRT_OF_2 = 0.7071067811865475f;
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        output.data[i] = 0.5 * x * (1.0 + erff(x * INV_SQRT_OF_2));
    }
    return output;
}

void gelu_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    const float INV_SQRT_OF_2 = 0.7071067811865475f;
    const float INV_SQRT_OF_2PI = 0.3989422804014327f;

    for(size_t i = 0; i < input.total_elements; i++) {
        float x = input.data[i];
        float cdf = 0.5 * (1.0 + erff(x * INV_SQRT_OF_2));
        float pdf = INV_SQRT_OF_2PI * expf(-0.5 * x * x);
        float dx = cdf + x * pdf;
        input_gradients->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor swish(const Tensor input) {
    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        if(x >= 0.0f) {
            float z = expf(-1.0 * x);
            output.data[i] = x * (1.0 / (1.0 + z));
        }
        else {
            float z = expf(x);
            output.data[i] = x * (z / (1.0 + z));
        }
    }
    return output;
}

void swish_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    for(size_t i = 0; i < input.total_elements; i++) {
        float x = input.data[i];
        if(x >= 0.0f) {
            float z = expf(-1.0 * x);
            float num = 1.0 / (1.0 + z);
            float dx = num + x * num * (1.0 - num);
            input_gradients->data[i] += upstream_gradients.data[i] * dx;
        }
        else {
            float z = expf(x);
            float num = z / (1.0 + z);
            float dx = num + x * num * (1.0 - num);
            input_gradients->data[i] += upstream_gradients.data[i] * dx;
        }
    }
}

Tensor mish(const Tensor input) {
    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        output.data[i] = x * tanhf(log1pf(expf(x)));
    }
    return output;
}

void mish_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float x = input.data[i];
        float z = tanhf(log1pf(expf(x)));
        float num;
        if(x >= 0.0) {
            float e = expf(-1.0 * x);
            num = 1.0 / (1.0 + e);
        }
        else {
            float e = expf(x);
            num = e / (1.0 + e);
        }

        float sech2 = 1.0 - z * z;
        float dx = z + x * num * sech2;
        input_gradients->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor gelu_sine(const Tensor input) {
    const float INV_SQRT_OF_2 = 0.7071067811865475f;

    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        float gelu_value = 0.5f * x * (1.0f + erff(x * INV_SQRT_OF_2));
        output.data[i] = gelu_value + 0.1f * sinf(x);
    }
    return output;
}
void gelu_sine_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    const float INV_SQRT_OF_2 = 0.7071067811865475f;
    const float INV_SQRT_OF_2PI = 0.3989422804014327f;
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float x = input.data[i];
        float cdf = 0.5f * (1.0f + erff(x * INV_SQRT_OF_2));
        float pdf = INV_SQRT_OF_2PI * expf(-0.5f * x * x);
        float dx = (cdf + x * pdf) + 0.1f * cosf(x);
        input_gradients->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor gelu_sinc(const Tensor input) {
    const float PI = 3.1415926535f;
    const float INV_SQRT_OF_2 = 0.7071067811865475f;

    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        float gelu_value = 0.5f * x * (1.0f + erff(x * INV_SQRT_OF_2));
        float sinc_value;
        if(fabsf(x) < 0.0001f) {
            sinc_value = 1.0f;
        }
        else {
            sinc_value = sinf(PI * x) / (PI * x);
        }
        output.data[i] = gelu_value * (1.0f + 0.5f * sinc_value);
    }
    return output;
}
void gelu_sinc_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    const float PI = 3.1415926535f;
    const float INV_SQRT_OF_2 = 0.7071067811865475f;
    const float INV_SQRT_OF_2PI = 0.3989422804014327f;

    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float x = input.data[i];
        float cdf = 0.5f * (1.0f + erff(x * INV_SQRT_OF_2));
        float pdf = INV_SQRT_OF_2PI * expf(-0.5f * x * x);
        float gelu_value = x * cdf;
        float gelu_derivative = cdf + x * pdf;
        float sinc_value;
        float sinc_derivative;
        if(fabsf(x) < 0.0001f) {
            sinc_value = 1.0f;
            sinc_derivative = 0.0f;
        }
        else {
            sinc_value = sinf(PI * x) / (PI * x);
            sinc_derivative = (PI * x * cosf(PI * x) - sinf(PI * x)) / (PI * x * x);
        }
        float factor = 1.0f + 0.5f * sinc_value;
        float dx = gelu_derivative * factor + gelu_value * 0.5f * sinc_derivative;
        input_gradients->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor mish_sine(const Tensor input) {
    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        float mish_value = x * tanhf(log1pf(expf(x)));
        output.data[i] = mish_value + 0.1f * sinf(x);
    }
    return output;
}
void mish_sine_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float x = input.data[i];
        float z = tanhf(log1pf(expf(x)));
        float num;
        if(x >= 0.0f) {
            float e = expf(-1.0f * x);
            num = 1.0f / (1.0f + e);
        }
        else {
            float e = expf(x);
            num = e / (1.0f + e);
        }
        float sech2 = 1.0f - z * z;
        float dx = (z + x * num * sech2) + 0.1f * cosf(x);
        input_gradients->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor mish_sinc(const Tensor input) {
    const float PI = 3.1415926535f;

    Tensor output = create_tensor(input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = input.data[i];
        float mish_value = x * tanhf(log1pf(expf(x)));
        float sinc_value;
        if(fabsf(x) < 0.0001f) {
            sinc_value = 1.0f;
        }
        else {
            sinc_value = sinf(PI * x) / (PI * x);
        }
        output.data[i] = mish_value * (1.0f + 0.5f * sinc_value);
    }
    return output;
}
void mish_sinc_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients) {
    const float PI = 3.1415926535f;

    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float x = input.data[i];
        float z = tanhf(log1pf(expf(x)));
        float num;
        if(x >= 0.0f) {
            float e = expf(-1.0f * x);
            num = 1.0f / (1.0f + e);
        }
        else {
            float e = expf(x);
            num = e / (1.0f + e);
        }
        float sech2 = 1.0f - z * z;
        float mish_value = x * z;
        float mish_derivative = z + x * num * sech2;
        float sinc_value;
        float sinc_derivative;
        if(fabsf(x) < 0.0001f) {
            sinc_value = 1.0f;
            sinc_derivative = 0.0f;
        }
        else {
            sinc_value = sinf(PI * x) / (PI * x);
            sinc_derivative = (PI * x * cosf(PI * x) - sinf(PI * x)) / (PI * x * x);
        }
        float factor = 1.0f + 0.5f * sinc_value;
        float dx = mish_derivative * factor + mish_value * 0.5f * sinc_derivative;
        input_gradients->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor categorical_cross_entropy(const Tensor prediction_input, const Tensor target_input) {
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_dims[i] = 1; }
    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zero(&output);

    const size_t CLASS_DIMENSION = TENSOR_DIMS - 1;
    size_t num_classes = prediction_input.dims[CLASS_DIMENSION];

    size_t class_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { class_indices[i] = 0; }
    size_t prediction_base = get_data_index(prediction_input, class_indices);
    size_t target_base = get_data_index(target_input, class_indices);

    float max_value = prediction_input.data[prediction_base];
    for(size_t i = 0; i < num_classes; i++) {
        if(prediction_input.data[prediction_base + i] > max_value) {
            max_value = prediction_input.data[prediction_base + i];
        }
    }

    float exp_sum = 0.0f;
    for(size_t i = 0; i < num_classes; i++) {
        float x = prediction_input.data[prediction_base + i] - max_value;
        exp_sum += expf(x);
    }
    float log_sum_exp = logf(exp_sum);

    float sample_loss = 0.0f;
    for(size_t i = 0; i < num_classes; i++) {
        float log_prob = (prediction_input.data[prediction_base + i] - max_value) - log_sum_exp;
        float target_prob = target_input.data[target_base + i];
        sample_loss += (-1.0f * target_prob) * log_prob;
    }

    output.data[0] = sample_loss;
    return output;
}

void categorical_cross_entropy_gradients(const Tensor prediction_input, const Tensor target_input, const Tensor upstream_gradients, Tensor* prediction_input_gradients, Tensor* target_input_gradients) {
    const size_t CLASS_DIMENSION = TENSOR_DIMS - 1;
    size_t num_classes = prediction_input.dims[CLASS_DIMENSION];

    float upstream_gradient = upstream_gradients.data[0];
    float reduction_scale = upstream_gradient;

    size_t class_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { class_indices[i] = 0; }
    size_t prediction_base = get_data_index(prediction_input, class_indices);
    size_t target_base = get_data_index(target_input, class_indices);
    size_t prediction_grad_base = get_data_index(*prediction_input_gradients, class_indices);
    size_t target_grad_base = get_data_index(*target_input_gradients, class_indices);

    float max_value = prediction_input.data[prediction_base];
    for(size_t i = 0; i < num_classes; i++) {
        if(prediction_input.data[prediction_base + i] > max_value) {
            max_value = prediction_input.data[prediction_base + i];
        }
    }

    float exp_sum = 0.0f;
    for(size_t i = 0; i < num_classes; i++) {
        float x = prediction_input.data[prediction_base + i] - max_value;
        exp_sum += expf(x);
    }
    float inverse_exp_sum = 1.0f / exp_sum;
    float log_sum_exp = logf(exp_sum);

    for(size_t i = 0; i < num_classes; i++) {
        float log_prob = (prediction_input.data[prediction_base + i] - max_value) - log_sum_exp;
        float prob = expf(prediction_input.data[prediction_base + i] - max_value) * inverse_exp_sum;
        float target_prob = target_input.data[target_base + i];

        prediction_input_gradients->data[prediction_grad_base + i] += (prob - target_prob) * reduction_scale;
        target_input_gradients->data[target_grad_base + i] += (-1.0f * log_prob) * reduction_scale;
    }
}

Tensor mean_square_error(const Tensor prediction_input, const Tensor target_input) {
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_dims[i] = 1; }
    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zero(&output);

    float sum = 0.0f;
    for(size_t i = 0; i < prediction_input.total_elements; i++) {
        float x = prediction_input.data[i] - target_input.data[i];
        sum += x * x;
    }
    output.data[0] = sum / (float)prediction_input.total_elements;
    return output;
}

void mean_square_error_gradients(const Tensor prediction_input, const Tensor target_input, const Tensor upstream_gradients, Tensor* prediction_input_gradients, Tensor* target_input_gradients) {
    float upstream_gradient = upstream_gradients.data[0];
    float scale = (2.0f * upstream_gradient) / (float)prediction_input.total_elements;

    for(size_t i = 0; i < prediction_input.total_elements; i++) {
        float difference = prediction_input.data[i] - target_input.data[i];
        prediction_input_gradients->data[i] += scale * difference;
        target_input_gradients->data[i] -= scale * difference;
    }
}

Tensor kl_divergence(const Tensor prediction_input, const Tensor target_input) {
    size_t output_dims[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { output_dims[i] = 1; }
    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zero(&output);

    const size_t CLASS_DIMENSION = TENSOR_DIMS - 1;
    size_t num_classes = prediction_input.dims[CLASS_DIMENSION];

    size_t class_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { class_indices[i] = 0; }
    size_t prediction_base = get_data_index(prediction_input, class_indices);
    size_t target_base = get_data_index(target_input, class_indices);

    float max_value = prediction_input.data[prediction_base];
    for(size_t i = 0; i < num_classes; i++) {
        if(prediction_input.data[prediction_base + i] > max_value) {
            max_value = prediction_input.data[prediction_base + i];
        }
    }

    float exp_sum = 0.0f;
    for(size_t i = 0; i < num_classes; i++) {
        float x = prediction_input.data[prediction_base + i] - max_value;
        exp_sum += expf(x);
    }
    float log_sum_exp = logf(exp_sum);

    float sample_loss = 0.0f;
    for(size_t i = 0; i < num_classes; i++) {
        float log_predicted_prob = (prediction_input.data[prediction_base + i] - max_value) - log_sum_exp;
        float target_prob = target_input.data[target_base + i];
        float target_entropy_term = (target_prob > 0.0f) ? (target_prob * logf(target_prob)) : 0.0f;
        sample_loss += target_entropy_term - (target_prob * log_predicted_prob);
    }

    output.data[0] = sample_loss;
    return output;
}

void kl_divergence_gradients(const Tensor prediction_input, const Tensor target_input, const Tensor upstream_gradients, Tensor* prediction_input_gradients, Tensor* target_input_gradients) {
    const size_t CLASS_DIMENSION = TENSOR_DIMS - 1;
    size_t num_classes = prediction_input.dims[CLASS_DIMENSION];

    float upstream_gradient = upstream_gradients.data[0];
    float reduction_scale = upstream_gradient;

    size_t class_indices[TENSOR_DIMS];
    for(size_t i = 0; i < TENSOR_DIMS; i++) { class_indices[i] = 0; }
    size_t prediction_base = get_data_index(prediction_input, class_indices);
    size_t target_base = get_data_index(target_input, class_indices);
    size_t prediction_grad_base = get_data_index(*prediction_input_gradients, class_indices);
    size_t target_grad_base = get_data_index(*target_input_gradients, class_indices);

    float max_value = prediction_input.data[prediction_base];
    for(size_t i = 0; i < num_classes; i++) {
        if(prediction_input.data[prediction_base + i] > max_value) {
            max_value = prediction_input.data[prediction_base + i];
        }
    }

    float exp_sum = 0.0f;
    for(size_t i = 0; i < num_classes; i++) {
        float x = prediction_input.data[prediction_base + i] - max_value;
        exp_sum += expf(x);
    }
    float inverse_exp_sum = 1.0f / exp_sum;
    float log_sum_exp = logf(exp_sum);

    for(size_t i = 0; i < num_classes; i++) {
        float log_predicted_prob = (prediction_input.data[prediction_base + i] - max_value) - log_sum_exp;
        float predicted_prob = expf(prediction_input.data[prediction_base + i] - max_value) * inverse_exp_sum;
        float target_prob = target_input.data[target_base + i];

        prediction_input_gradients->data[prediction_grad_base + i] += (predicted_prob - target_prob) * reduction_scale;

        float target_entropy_gradient = (target_prob > 0.0f) ? (logf(target_prob) + 1.0f) : 0.0f;
        target_input_gradients->data[target_grad_base + i] += (target_entropy_gradient - log_predicted_prob) * reduction_scale;
    }
}

Tensor dropout(const Tensor input, float drop_probability, uint64_t seed) {
    Tensor output = create_tensor(input.dims);
    float keep_probability = 1.0f - drop_probability;
    float scale = (keep_probability > 0.0f) ? (1.0f / keep_probability) : 0.0f;
    for(size_t i = 0; i < output.total_elements; i++) {
        uint64_t rng_state = seed ^ ((uint64_t)i * 6364136223846793005ULL);
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t rng_xorshifted = ((rng_state >> 18u) ^ rng_state) >> 27u;
        uint32_t rng_rotation = rng_state >> 59u;
        uint32_t rng_bits = (rng_xorshifted >> rng_rotation) | (rng_xorshifted << ((-rng_rotation) & 31));
        float uniform_sample = ldexpf((float)(rng_bits >> 8), -24);

        if(uniform_sample < drop_probability) {
            output.data[i] = 0.0f;
        }
        else {
            output.data[i] = input.data[i] * scale;
        }
    }
    return output;
}

void dropout_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients, float drop_probability, uint64_t seed) {
    float keep_probability = 1.0f - drop_probability;
    float scale = (keep_probability > 0.0f) ? (1.0f / keep_probability) : 0.0f;
    for(size_t i = 0; i < input.total_elements; i++) {
        uint64_t rng_state = seed ^ ((uint64_t)i * 6364136223846793005ULL);
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t rng_xorshifted = ((rng_state >> 18u) ^ rng_state) >> 27u;
        uint32_t rng_rotation = rng_state >> 59u;
        uint32_t rng_bits = (rng_xorshifted >> rng_rotation) | (rng_xorshifted << ((-rng_rotation) & 31));
        float uniform_sample = ldexpf((float)(rng_bits >> 8), -24);

        if(uniform_sample < drop_probability) { continue; }
        input_gradients->data[i] += upstream_gradients.data[i] * scale;
    }
}

Tensor droppath(const Tensor input, float drop_probability, size_t sample_dimension, uint64_t seed) {
    Tensor output = create_tensor(input.dims);
    float keep_probability = 1.0f - drop_probability;
    float scale = (keep_probability > 0.0f) ? (1.0f / keep_probability) : 0.0f;

    size_t sample_stride = 1;
    for(size_t i = sample_dimension + 1; i < TENSOR_DIMS; i++) {
        sample_stride *= input.dims[i];
    }
    size_t total_samples = input.dims[sample_dimension];

    for(size_t i = 0; i < output.total_elements; i++) {
        size_t sample_coordinate = (i / sample_stride) % total_samples;
        uint64_t rng_state = seed ^ ((uint64_t)sample_coordinate * 6364136223846793005ULL);
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t rng_xorshifted = ((rng_state >> 18u) ^ rng_state) >> 27u;
        uint32_t rng_rotation = rng_state >> 59u;
        uint32_t rng_bits = (rng_xorshifted >> rng_rotation) | (rng_xorshifted << ((-rng_rotation) & 31));
        float uniform_sample = ldexpf((float)(rng_bits >> 8), -24);

        if(uniform_sample < drop_probability) {
            output.data[i] = 0.0f;
        }
        else {
            output.data[i] = input.data[i] * scale;
        }
    }
    return output;
}

void droppath_gradients(const Tensor input, const Tensor upstream_gradients, Tensor* input_gradients, float drop_probability, size_t sample_dimension, uint64_t seed) {
    float keep_probability = 1.0f - drop_probability;
    float scale = (keep_probability > 0.0f) ? (1.0f / keep_probability) : 0.0f;

    size_t sample_stride = 1;
    for(size_t i = sample_dimension + 1; i < TENSOR_DIMS; i++) {
        sample_stride *= input.dims[i];
    }
    size_t total_samples = input.dims[sample_dimension];

    for(size_t i = 0; i < input.total_elements; i++) {
        size_t sample_coordinate = (i / sample_stride) % total_samples;
        uint64_t rng_state = seed ^ ((uint64_t)sample_coordinate * 6364136223846793005ULL);
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t rng_xorshifted = ((rng_state >> 18u) ^ rng_state) >> 27u;
        uint32_t rng_rotation = rng_state >> 59u;
        uint32_t rng_bits = (rng_xorshifted >> rng_rotation) | (rng_xorshifted << ((-rng_rotation) & 31));
        float uniform_sample = ldexpf((float)(rng_bits >> 8), -24);

        if(uniform_sample < drop_probability) { continue; }
        input_gradients->data[i] += upstream_gradients.data[i] * scale;
    }
}

Tensor zoneout(const Tensor current_input, const Tensor previous_input, float zoneout_probability, uint64_t seed) {
    Tensor output = create_tensor(current_input.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        uint64_t rng_state = seed ^ ((uint64_t)i * 6364136223846793005ULL);
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t rng_xorshifted = ((rng_state >> 18u) ^ rng_state) >> 27u;
        uint32_t rng_rotation = rng_state >> 59u;
        uint32_t rng_bits = (rng_xorshifted >> rng_rotation) | (rng_xorshifted << ((-rng_rotation) & 31));
        float uniform_sample = ldexpf((float)(rng_bits >> 8), -24);

        if(uniform_sample < zoneout_probability) {
            output.data[i] = previous_input.data[i];
        }
        else {
            output.data[i] = current_input.data[i];
        }
    }
    return output;
}

void zoneout_gradients(const Tensor current_input, const Tensor upstream_gradients, Tensor* current_input_gradients, Tensor* previous_input_gradients, float zoneout_probability, uint64_t seed) {
    for(size_t i = 0; i < current_input.total_elements; i++) {
        uint64_t rng_state = seed ^ ((uint64_t)i * 6364136223846793005ULL);
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t rng_xorshifted = ((rng_state >> 18u) ^ rng_state) >> 27u;
        uint32_t rng_rotation = rng_state >> 59u;
        uint32_t rng_bits = (rng_xorshifted >> rng_rotation) | (rng_xorshifted << ((-rng_rotation) & 31));
        float uniform_sample = ldexpf((float)(rng_bits >> 8), -24);

        if(uniform_sample < zoneout_probability) {
            previous_input_gradients->data[i] += upstream_gradients.data[i];
        }
        else {
            current_input_gradients->data[i] += upstream_gradients.data[i];
        }
    }
}

DirectedAcyclicGraph create_directed_acyclic_graph() {
    DirectedAcyclicGraph dag;
    dag.current_time_step = 0;
    dag.total_nodes = 0;
    dag.total_tape_entries = 0;
    dag.nodes = NULL;
    dag.tape_entries = NULL;
    return dag;
}

size_t add_node(DirectedAcyclicGraph* dag, OperationType op_type, const OperationConfiguration* op_config) {
    size_t node_index = dag->total_nodes;
    dag->total_nodes++;
    dag->nodes = realloc(dag->nodes, dag->total_nodes * sizeof(Node));

    dag->nodes[node_index].op_type = op_type;
    if(op_config != NULL) {
        dag->nodes[node_index].op_config = *op_config;
    }
    dag->nodes[node_index].total_src_nodes = 0;
    dag->nodes[node_index].total_dst_nodes = 0;
    dag->nodes[node_index].src_node_indices = NULL;
    dag->nodes[node_index].dst_node_indices = NULL;
    dag->nodes[node_index].total_tape_entries = 0;
    dag->nodes[node_index].tape_entry_indices = NULL;

    return node_index;
}

void add_edge(DirectedAcyclicGraph* dag, size_t src_node_index, size_t dst_node_index) {
    dag->nodes[src_node_index].total_dst_nodes++;
    dag->nodes[src_node_index].dst_node_indices = realloc(dag->nodes[src_node_index].dst_node_indices, dag->nodes[src_node_index].total_dst_nodes * sizeof(size_t));
    dag->nodes[src_node_index].dst_node_indices[dag->nodes[src_node_index].total_dst_nodes - 1] = dst_node_index;

    dag->nodes[dst_node_index].total_src_nodes++;
    dag->nodes[dst_node_index].src_node_indices = realloc(dag->nodes[dst_node_index].src_node_indices, dag->nodes[dst_node_index].total_src_nodes * sizeof(size_t));
    dag->nodes[dst_node_index].src_node_indices[dag->nodes[dst_node_index].total_src_nodes - 1] = src_node_index;
}

void clear_memory(DirectedAcyclicGraph* dag) {
    size_t new_tape_entry_index = 0;
    size_t new_tape_entries_arr_capacity = dag->total_nodes;
    TapeEntry* new_tape_entries = malloc(new_tape_entries_arr_capacity * sizeof(TapeEntry));

    for(size_t i = 0; i < dag->total_nodes; i++) {
        free(dag->nodes[i].tape_entry_indices);
        dag->nodes[i].tape_entry_indices = NULL;
        dag->nodes[i].total_tape_entries = 0;
    }

    for(size_t tape_entry_index = 0; tape_entry_index < dag->total_tape_entries; tape_entry_index++) {
        size_t node_index = dag->tape_entries[tape_entry_index].node_index;
        if(dag->nodes[node_index].op_type == OPERATION_TYPE_PARAMETERS) {
            new_tape_entries[new_tape_entry_index].time_step = 0;
            new_tape_entries[new_tape_entry_index].node_index = node_index;
            new_tape_entries[new_tape_entry_index].total_src_tape_entries = 0;
            new_tape_entries[new_tape_entry_index].total_dst_tape_entries = 0;
            new_tape_entries[new_tape_entry_index].src_tape_entry_indices = NULL;
            new_tape_entries[new_tape_entry_index].dst_tape_entry_indices = NULL;
            new_tape_entries[new_tape_entry_index].output_tensor = dag->tape_entries[tape_entry_index].output_tensor;
            dag->tape_entries[tape_entry_index].output_tensor.data = NULL;
            new_tape_entries[new_tape_entry_index].gradient_tensor = create_tensor(NULL);

            dag->nodes[node_index].total_tape_entries = 1;
            dag->nodes[node_index].tape_entry_indices = malloc(sizeof(size_t));
            dag->nodes[node_index].tape_entry_indices[0] = new_tape_entry_index;
            new_tape_entry_index++;
        }
    }
    for(size_t i = 0; i < dag->total_tape_entries; i++) {
        free(dag->tape_entries[i].src_tape_entry_indices);
        free(dag->tape_entries[i].dst_tape_entry_indices);
        free_tensor(&dag->tape_entries[i].output_tensor);
        free_tensor(&dag->tape_entries[i].gradient_tensor);
    }

    new_tape_entries_arr_capacity = new_tape_entry_index;
    new_tape_entries = realloc(new_tape_entries, new_tape_entries_arr_capacity * sizeof(TapeEntry));
    free(dag->tape_entries);
    dag->total_tape_entries = new_tape_entries_arr_capacity;
    dag->tape_entries = new_tape_entries;
}

void initialize_parameters(DirectedAcyclicGraph* dag) {
    const float SQRT2 = 1.41421356237f;
    const float NORM_THRESHOLD = 1e-6f;

    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        if(dag->nodes[node_index].op_type != OPERATION_TYPE_PARAMETERS) { continue; }
        if(dag->nodes[node_index].total_tape_entries == 0) { continue; }

        size_t tape_entry_index = dag->nodes[node_index].tape_entry_indices[0];
        Tensor* parameter_tensor = &dag->tape_entries[tape_entry_index].output_tensor;
        if(parameter_tensor->data == NULL || parameter_tensor->total_elements == 0) { continue; }

        size_t num_rows = parameter_tensor->dims[TENSOR_DIMS - 2];
        size_t num_columns = parameter_tensor->dims[TENSOR_DIMS - 1];

        bool only_used_in_addition = true;
        bool only_used_in_matmul = true;
        bool only_used_in_conv = true;

        size_t closest_relu_like_function = SIZE_MAX;
        size_t closest_softmax = SIZE_MAX;
        size_t closest_cce_loss = SIZE_MAX;

        for(size_t i = 0; i < dag->nodes[node_index].total_dst_nodes; i++) {
            size_t depth_one_node_index = dag->nodes[node_index].dst_node_indices[i];

            if(dag->nodes[depth_one_node_index].op_type != OPERATION_TYPE_ADD) { only_used_in_addition = false; }
            if(dag->nodes[depth_one_node_index].op_type != OPERATION_TYPE_MATRIX_MULTIPLICATION) { only_used_in_matmul = false; }
            if(dag->nodes[depth_one_node_index].op_type != OPERATION_TYPE_CONVOLUTION) { only_used_in_conv = false; }

            OperationType depth_one_op = dag->nodes[depth_one_node_index].op_type;
            if((depth_one_op == OPERATION_TYPE_GELU || depth_one_op == OPERATION_TYPE_SWISH || depth_one_op == OPERATION_TYPE_MISH) && 1 < closest_relu_like_function) { closest_relu_like_function = 1; }
            else if(depth_one_op == OPERATION_TYPE_SOFTMAX && 1 < closest_softmax) { closest_softmax = 1; }
            else if(depth_one_op == OPERATION_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS && 1 < closest_cce_loss) { closest_cce_loss = 1; }

            for(size_t j = 0; j < dag->nodes[depth_one_node_index].total_dst_nodes; j++) {
                size_t depth_two_node_index = dag->nodes[depth_one_node_index].dst_node_indices[j];

                OperationType depth_two_op = dag->nodes[depth_two_node_index].op_type;
                if((depth_two_op == OPERATION_TYPE_GELU || depth_two_op == OPERATION_TYPE_SWISH || depth_two_op == OPERATION_TYPE_MISH) && 2 < closest_relu_like_function) { closest_relu_like_function = 2; }
                else if(depth_two_op == OPERATION_TYPE_SOFTMAX && 2 < closest_softmax) { closest_softmax = 2; }
                else if(depth_two_op == OPERATION_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS && 2 < closest_cce_loss) { closest_cce_loss = 2; }

                for(size_t k = 0; k < dag->nodes[depth_two_node_index].total_dst_nodes; k++) {
                    size_t depth_three_node_index = dag->nodes[depth_two_node_index].dst_node_indices[k];

                    OperationType depth_three_op = dag->nodes[depth_three_node_index].op_type;
                    if((depth_three_op == OPERATION_TYPE_GELU || depth_three_op == OPERATION_TYPE_SWISH || depth_three_op == OPERATION_TYPE_MISH) && 3 < closest_relu_like_function) { closest_relu_like_function = 3; }
                    else if(depth_three_op == OPERATION_TYPE_SOFTMAX && 3 < closest_softmax) { closest_softmax = 3; }
                    else if(depth_three_op == OPERATION_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS && 3 < closest_cce_loss) { closest_cce_loss = 3; }
                }
            }
        }

        set_tensor_data_to_zero(parameter_tensor);

        if(only_used_in_addition) {
            continue;
        }
        else {
            float fan_in = (float)num_rows;
            float fan_out = (float)num_columns;

            if(only_used_in_conv) {
                size_t out_channels = parameter_tensor->dims[0];
                size_t in_channels = parameter_tensor->dims[1];
                size_t kernel_extent = parameter_tensor->dims[TENSOR_DIMS - 1];
                fan_in = (float)in_channels * (float)kernel_extent;
                fan_out = (float)out_channels * (float)kernel_extent;
            }

            float std_dev;
            if(closest_relu_like_function != SIZE_MAX) {
                std_dev = SQRT2 / sqrtf(fan_in);
            }
            else {
                std_dev = sqrtf(2.0f / (fan_in + fan_out));
            }

            if((closest_softmax != SIZE_MAX) || (closest_cce_loss != SIZE_MAX)) {
                std_dev = sqrtf(1.0f / fan_in);
            }

            add_gaussian_noise_to_tensor(parameter_tensor, 0.0, std_dev);

            if(only_used_in_matmul && (num_rows >= num_columns)) {
                make_columns_orthonormal_householder(parameter_tensor, NORM_THRESHOLD);
            }
        }
    }
}

void emergence_promoting_initialization(DirectedAcyclicGraph* dag, float alpha) {
    if(dag->total_nodes == 0) { return; }

    size_t* node_depths = malloc(dag->total_nodes * sizeof(size_t));
    if(node_depths == NULL) { return; }
    for(size_t i = 0; i < dag->total_nodes; i++) {
        node_depths[i] = 0;
    }

    bool depths_changed = true;
    while(depths_changed) {
        depths_changed = false;
        for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
            for(size_t i = 0; i < dag->nodes[node_index].total_dst_nodes; i++) {
                size_t dst_node_index = dag->nodes[node_index].dst_node_indices[i];
                if(node_depths[dst_node_index] < (node_depths[node_index] + 1)) {
                    node_depths[dst_node_index] = node_depths[node_index] + 1;
                    depths_changed = true;
                }
            }
        }
    }

    size_t total_weight_parameters = 0;
    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        if(dag->nodes[node_index].op_type != OPERATION_TYPE_PARAMETERS) { continue; }
        for(size_t i = 0; i < dag->nodes[node_index].total_dst_nodes; i++) {
            OperationType dst_op_type = dag->nodes[dag->nodes[node_index].dst_node_indices[i]].op_type;
            if(dst_op_type == OPERATION_TYPE_MATRIX_MULTIPLICATION || dst_op_type == OPERATION_TYPE_CONVOLUTION) {
                total_weight_parameters++;
                break;
            }
        }
    }

    if(total_weight_parameters == 0) {
        free(node_depths);
        return;
    }

    size_t* weight_parameter_nodes = malloc(total_weight_parameters * sizeof(size_t));
    size_t* weight_parameter_depths = malloc(total_weight_parameters * sizeof(size_t));
    if(weight_parameter_nodes == NULL || weight_parameter_depths == NULL) {
        free(node_depths);
        free(weight_parameter_nodes);
        free(weight_parameter_depths);
        return;
    }

    size_t weight_parameter_count = 0;
    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        if(dag->nodes[node_index].op_type != OPERATION_TYPE_PARAMETERS) { continue; }

        bool consumed_by_weight_operation = false;
        size_t consuming_operation_depth = 0;
        for(size_t i = 0; i < dag->nodes[node_index].total_dst_nodes; i++) {
            size_t dst_node_index = dag->nodes[node_index].dst_node_indices[i];
            OperationType dst_op_type = dag->nodes[dst_node_index].op_type;
            if(dst_op_type == OPERATION_TYPE_MATRIX_MULTIPLICATION || dst_op_type == OPERATION_TYPE_CONVOLUTION) {
                if(!consumed_by_weight_operation || node_depths[dst_node_index] < consuming_operation_depth) {
                    consuming_operation_depth = node_depths[dst_node_index];
                }
                consumed_by_weight_operation = true;
            }
        }

        if(consumed_by_weight_operation) {
            weight_parameter_nodes[weight_parameter_count] = node_index;
            weight_parameter_depths[weight_parameter_count] = consuming_operation_depth;
            weight_parameter_count++;
        }
    }

    size_t total_weight_layers = 0;
    for(size_t i = 0; i < weight_parameter_count; i++) {
        bool first_occurrence = true;
        for(size_t j = 0; j < i; j++) {
            if(weight_parameter_depths[j] == weight_parameter_depths[i]) {
                first_occurrence = false;
                break;
            }
        }
        if(first_occurrence) { total_weight_layers++; }
    }

    size_t pivot_layer = total_weight_layers / 2;

    for(size_t i = 0; i < weight_parameter_count; i++) {
        size_t node_index = weight_parameter_nodes[i];
        if(dag->nodes[node_index].total_tape_entries == 0) { continue; }

        size_t tape_entry_index = dag->nodes[node_index].tape_entry_indices[0];
        Tensor* parameter_tensor = &dag->tape_entries[tape_entry_index].output_tensor;
        if(parameter_tensor->data == NULL || parameter_tensor->total_elements == 0) { continue; }

        size_t layer_index = 0;
        for(size_t j = 0; j < weight_parameter_count; j++) {
            if(weight_parameter_depths[j] >= weight_parameter_depths[i]) { continue; }
            bool first_occurrence = true;
            for(size_t k = 0; k < j; k++) {
                if(weight_parameter_depths[k] == weight_parameter_depths[j]) {
                    first_occurrence = false;
                    break;
                }
            }
            if(first_occurrence) { layer_index++; }
        }

        if(layer_index == pivot_layer) { continue; }

        if(layer_index < pivot_layer) {
            size_t exponent = pivot_layer - layer_index;
            float scaling_factor = 1.0f;
            for(size_t step = 0; step < exponent; step++) { scaling_factor *= alpha; }
            for(size_t element_index = 0; element_index < parameter_tensor->total_elements; element_index++) {
                parameter_tensor->data[element_index] /= scaling_factor;
            }
        }
        else {
            size_t exponent = layer_index - pivot_layer;
            float scaling_factor = 1.0f;
            for(size_t step = 0; step < exponent; step++) { scaling_factor *= alpha; }
            for(size_t element_index = 0; element_index < parameter_tensor->total_elements; element_index++) {
                parameter_tensor->data[element_index] *= scaling_factor;
            }
        }
    }

    free(node_depths);
    free(weight_parameter_nodes);
    free(weight_parameter_depths);
}

void add_input_to_dag(DirectedAcyclicGraph* dag, size_t input_node_index, const Tensor input) {
    size_t tape_entry_index = dag->total_tape_entries;
    dag->total_tape_entries++;
    dag->tape_entries = realloc(dag->tape_entries, dag->total_tape_entries * sizeof(TapeEntry));

    dag->tape_entries[tape_entry_index].time_step = (dag->current_time_step + 1);
    dag->tape_entries[tape_entry_index].node_index = input_node_index;
    dag->tape_entries[tape_entry_index].total_src_tape_entries = 0;
    dag->tape_entries[tape_entry_index].total_dst_tape_entries = 0;
    dag->tape_entries[tape_entry_index].src_tape_entry_indices = NULL;
    dag->tape_entries[tape_entry_index].dst_tape_entry_indices = NULL;
    dag->tape_entries[tape_entry_index].output_tensor = copy_tensor(input);
    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(NULL);

    size_t tape_entry_index_inside_node = dag->nodes[input_node_index].total_tape_entries;
    dag->nodes[input_node_index].total_tape_entries++;
    dag->nodes[input_node_index].tape_entry_indices = realloc(dag->nodes[input_node_index].tape_entry_indices, dag->nodes[input_node_index].total_tape_entries * sizeof(size_t));
    dag->nodes[input_node_index].tape_entry_indices[tape_entry_index_inside_node] = tape_entry_index;
}

bool should_add_tape_entry(const DirectedAcyclicGraph dag, size_t node_index) {
    if(dag.nodes[node_index].total_tape_entries > 0) {
        size_t tape_entry_index_inside_node = dag.nodes[node_index].total_tape_entries - 1;
        size_t tape_entry_index = dag.nodes[node_index].tape_entry_indices[tape_entry_index_inside_node];
        if(dag.tape_entries[tape_entry_index].time_step == dag.current_time_step) { return false; }
    }

    for(size_t i = 0; i < dag.nodes[node_index].total_src_nodes; i++) {
        size_t src_node_index = dag.nodes[node_index].src_node_indices[i];
        if(dag.nodes[src_node_index].total_tape_entries > 0) {
            if(dag.nodes[src_node_index].op_type == OPERATION_TYPE_PARAMETERS) { continue; }
            size_t src_tape_entry_index = dag.nodes[src_node_index].tape_entry_indices[dag.nodes[src_node_index].total_tape_entries - 1];
            if(dag.tape_entries[src_tape_entry_index].time_step != dag.current_time_step) { return false; }
        }
        else { return false; }
    }
    return true;
}

void add_tape_entry(DirectedAcyclicGraph* dag, size_t node_index) {
    size_t tape_entry_index = dag->total_tape_entries;
    dag->total_tape_entries++;
    dag->tape_entries = realloc(dag->tape_entries, dag->total_tape_entries * sizeof(TapeEntry));

    dag->tape_entries[tape_entry_index].time_step = dag->current_time_step;
    dag->tape_entries[tape_entry_index].node_index = node_index;
    dag->tape_entries[tape_entry_index].total_src_tape_entries = dag->nodes[node_index].total_src_nodes;
    dag->tape_entries[tape_entry_index].total_dst_tape_entries = 0;
    dag->tape_entries[tape_entry_index].src_tape_entry_indices = malloc(dag->tape_entries[tape_entry_index].total_src_tape_entries * sizeof(size_t));
    dag->tape_entries[tape_entry_index].dst_tape_entry_indices = NULL;

    for(size_t i = 0; i < dag->tape_entries[tape_entry_index].total_src_tape_entries; i++) {
        size_t src_node_index = dag->nodes[node_index].src_node_indices[i];
        size_t src_tape_entry_index = dag->nodes[src_node_index].tape_entry_indices[dag->nodes[src_node_index].total_tape_entries - 1];
        dag->tape_entries[tape_entry_index].src_tape_entry_indices[i] = src_tape_entry_index;

        size_t dst_tape_entry_index_inside_src_tape_entry = dag->tape_entries[src_tape_entry_index].total_dst_tape_entries;
        dag->tape_entries[src_tape_entry_index].total_dst_tape_entries++;
        dag->tape_entries[src_tape_entry_index].dst_tape_entry_indices = realloc(dag->tape_entries[src_tape_entry_index].dst_tape_entry_indices, dag->tape_entries[src_tape_entry_index].total_dst_tape_entries * sizeof(size_t));
        dag->tape_entries[src_tape_entry_index].dst_tape_entry_indices[dst_tape_entry_index_inside_src_tape_entry] = tape_entry_index;
    }
    dag->tape_entries[tape_entry_index].output_tensor = create_tensor(NULL);
    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(NULL);

    size_t tape_entry_index_inside_node = dag->nodes[node_index].total_tape_entries;
    dag->nodes[node_index].total_tape_entries++;
    dag->nodes[node_index].tape_entry_indices = realloc(dag->nodes[node_index].tape_entry_indices, dag->nodes[node_index].total_tape_entries * sizeof(size_t));
    dag->nodes[node_index].tape_entry_indices[tape_entry_index_inside_node] = tape_entry_index;
}

static inline void forward(DirectedAcyclicGraph* dag) {
    dag->current_time_step++;

    size_t tape_entry_index = 0;
    while(tape_entry_index < dag->total_tape_entries) {
        if(dag->tape_entries[tape_entry_index].time_step == dag->current_time_step) {
            size_t node_index = dag->tape_entries[tape_entry_index].node_index;
            switch(dag->nodes[node_index].op_type) {
                case OPERATION_TYPE_RESIZE: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    ResizeConfiguration* resize_config = &dag->nodes[node_index].op_config.resize;
                    dag->tape_entries[tape_entry_index].output_tensor = resize(dag->tape_entries[src_tape_entry_index].output_tensor, resize_config->new_dims);
                    break;
                }
                case OPERATION_TYPE_CONCATENATE: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    ConcatenationConfiguration* concatenation_config = &dag->nodes[node_index].op_config.concatenation;
                    dag->tape_entries[tape_entry_index].output_tensor = concatenate(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, concatenation_config->concatenation_dimension);
                    break;
                }
                case OPERATION_TYPE_ADD: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = add(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OPERATION_TYPE_SCALAR_MULTIPLICATION: {
                    size_t scalar_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t tensor_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = scalar_multiplication(dag->tape_entries[scalar_tape_entry_index].output_tensor, dag->tape_entries[tensor_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_ELEMENTWISE_MULTIPLICATION: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = elementwise_mul(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OPERATION_TYPE_MATRIX_MULTIPLICATION: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = matrix_multiplication(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OPERATION_TYPE_CONVOLUTION: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    ConvolutionConfiguration* convolution_config = &dag->nodes[node_index].op_config.convolution;
                    dag->tape_entries[tape_entry_index].output_tensor = convolution(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, convolution_config->padding, convolution_config->total_kernels);
                    break;
                }
                case OPERATION_TYPE_SIGMOID: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = sigmoid(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_SOFTMAX: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    SoftmaxConfiguration* softmax_config = &dag->nodes[node_index].op_config.softmax;
                    dag->tape_entries[tape_entry_index].output_tensor = softmax(dag->tape_entries[src_tape_entry_index].output_tensor, softmax_config->batch_dimension);
                    break;
                }
                case OPERATION_TYPE_GELU: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = gelu(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_SWISH: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = swish(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_MISH: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = mish(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_GELU_SINE: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = gelu_sine(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_GELU_SINC: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = gelu_sinc(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_MISH_SINE: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = mish_sine(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_MISH_SINC: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = mish_sinc(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_LAYER_NORM: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t gamma_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    size_t beta_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[2];
                    LayerNormConfiguration* layer_norm_config = &dag->nodes[node_index].op_config.layer_norm;
                    dag->tape_entries[tape_entry_index].output_tensor = layer_norm(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[gamma_tape_entry_index].output_tensor, dag->tape_entries[beta_tape_entry_index].output_tensor, layer_norm_config->epsilon, layer_norm_config->normalization_dimension);
                    break;
                }
                case OPERATION_TYPE_RMS_LAYER_NORM: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t gamma_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    LayerNormConfiguration* layer_norm_config = &dag->nodes[node_index].op_config.layer_norm;
                    dag->tape_entries[tape_entry_index].output_tensor = rms_norm(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[gamma_tape_entry_index].output_tensor, layer_norm_config->epsilon, layer_norm_config->normalization_dimension);
                    break;
                }
                case OPERATION_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS: {
                    size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = categorical_cross_entropy(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_MEAN_SQUARE_ERROR: {
                    size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = mean_square_error(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_KL_DIVERGENCE: {
                    size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = kl_divergence(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor);
                    break;
                }
                case OPERATION_TYPE_DROPOUT: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    StochasticRegularizationConfiguration* config = &dag->nodes[node_index].op_config.stochastic_regularization;
                    uint64_t seed = config->base_seed ^ (dag->tape_entries[tape_entry_index].time_step * 0x9e3779b97f4a7c15ULL) ^ ((uint64_t)node_index * 0xbf58476d1ce4e5b9ULL);
                    dag->tape_entries[tape_entry_index].output_tensor = dropout(dag->tape_entries[src_tape_entry_index].output_tensor, config->probability, seed);
                    break;
                }
                case OPERATION_TYPE_DROPPATH: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    StochasticRegularizationConfiguration* config = &dag->nodes[node_index].op_config.stochastic_regularization;
                    uint64_t seed = config->base_seed ^ (dag->tape_entries[tape_entry_index].time_step * 0x9e3779b97f4a7c15ULL) ^ ((uint64_t)node_index * 0xbf58476d1ce4e5b9ULL);
                    dag->tape_entries[tape_entry_index].output_tensor = droppath(dag->tape_entries[src_tape_entry_index].output_tensor, config->probability, config->sample_dimension, seed);
                    break;
                }
                case OPERATION_TYPE_ZONEOUT: {
                    size_t current_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t previous_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    StochasticRegularizationConfiguration* config = &dag->nodes[node_index].op_config.stochastic_regularization;
                    uint64_t seed = config->base_seed ^ (dag->tape_entries[tape_entry_index].time_step * 0x9e3779b97f4a7c15ULL) ^ ((uint64_t)node_index * 0xbf58476d1ce4e5b9ULL);
                    dag->tape_entries[tape_entry_index].output_tensor = zoneout(dag->tape_entries[current_tape_entry_index].output_tensor, dag->tape_entries[previous_tape_entry_index].output_tensor, config->probability, seed);
                    break;
                }
                default: { break; }
            }

            for(size_t i = 0; i < dag->nodes[node_index].total_dst_nodes; i++) {
                size_t dst_node_index = dag->nodes[node_index].dst_node_indices[i];
                if(should_add_tape_entry(*dag, dst_node_index)) {
                    add_tape_entry(dag, dst_node_index);
                }
            }
        }
        tape_entry_index++;
    }
}

void backward(DirectedAcyclicGraph* dag) {
    for(size_t tape_entry_index = dag->total_tape_entries; tape_entry_index-- > 0;) {
        for(size_t i = 0; i < dag->tape_entries[tape_entry_index].total_src_tape_entries; i++) {
            size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[i];
            if(dag->tape_entries[src_tape_entry_index].gradient_tensor.data == NULL) {
                dag->tape_entries[src_tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[src_tape_entry_index].output_tensor.dims);
                set_tensor_data_to_zero(&dag->tape_entries[src_tape_entry_index].gradient_tensor);
            }
        }
        size_t node_index = dag->tape_entries[tape_entry_index].node_index;
        switch(dag->nodes[node_index].op_type) {
            case OPERATION_TYPE_RESIZE: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                resize_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_CONCATENATE: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                ConcatenationConfiguration* concatenation_config = &dag->nodes[node_index].op_config.concatenation;
                concatenate_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor, concatenation_config->concatenation_dimension);
                break;
            }
            case OPERATION_TYPE_ADD: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                add_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_SCALAR_MULTIPLICATION: {
                size_t scalar_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t tensor_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                scalar_multiplication_gradients(dag->tape_entries[scalar_tape_entry_index].output_tensor, dag->tape_entries[tensor_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[scalar_tape_entry_index].gradient_tensor, &dag->tape_entries[tensor_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_ELEMENTWISE_MULTIPLICATION: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                elementwise_mul_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_MATRIX_MULTIPLICATION: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                matrix_multiplication_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_CONVOLUTION: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                ConvolutionConfiguration* convolution_config = &dag->nodes[node_index].op_config.convolution;
                convolution_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor, convolution_config->padding, convolution_config->total_kernels);
                break;
            }
            case OPERATION_TYPE_SIGMOID: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                sigmoid_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_SOFTMAX: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                SoftmaxConfiguration* softmax_config = &dag->nodes[node_index].op_config.softmax;
                softmax_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor, softmax_config->batch_dimension);
                break;
            }
            case OPERATION_TYPE_GELU: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                gelu_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_SWISH: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                swish_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_MISH: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                mish_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_GELU_SINE: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                gelu_sine_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_GELU_SINC: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                gelu_sinc_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_MISH_SINE: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                mish_sine_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_MISH_SINC: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                mish_sinc_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_LAYER_NORM: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t gamma_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                size_t beta_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[2];
                LayerNormConfiguration* layer_norm_config = &dag->nodes[node_index].op_config.layer_norm;
                layer_norm_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[gamma_tape_entry_index].output_tensor, dag->tape_entries[beta_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor, &dag->tape_entries[gamma_tape_entry_index].gradient_tensor, &dag->tape_entries[beta_tape_entry_index].gradient_tensor, layer_norm_config->epsilon, layer_norm_config->normalization_dimension);
                break;
            }
            case OPERATION_TYPE_RMS_LAYER_NORM: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t gamma_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                LayerNormConfiguration* layer_norm_config = &dag->nodes[node_index].op_config.layer_norm;
                rms_norm_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[gamma_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor, &dag->tape_entries[gamma_tape_entry_index].gradient_tensor, layer_norm_config->epsilon, layer_norm_config->normalization_dimension);
                break;
            }
            case OPERATION_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS: {
                if(dag->tape_entries[tape_entry_index].total_dst_tape_entries == 0 && dag->tape_entries[tape_entry_index].gradient_tensor.data == NULL) {
                    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[tape_entry_index].output_tensor.dims);
                    dag->tape_entries[tape_entry_index].gradient_tensor.data[0] = 1.0f;
                }
                size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                categorical_cross_entropy_gradients(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[prediction_tape_entry_index].gradient_tensor, &dag->tape_entries[target_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_MEAN_SQUARE_ERROR: {
                if(dag->tape_entries[tape_entry_index].total_dst_tape_entries == 0 && dag->tape_entries[tape_entry_index].gradient_tensor.data == NULL) {
                    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[tape_entry_index].output_tensor.dims);
                    dag->tape_entries[tape_entry_index].gradient_tensor.data[0] = 1.0f;
                }
                size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                mean_square_error_gradients(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[prediction_tape_entry_index].gradient_tensor, &dag->tape_entries[target_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_KL_DIVERGENCE: {
                if(dag->tape_entries[tape_entry_index].total_dst_tape_entries == 0 && dag->tape_entries[tape_entry_index].gradient_tensor.data == NULL) {
                    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[tape_entry_index].output_tensor.dims);
                    dag->tape_entries[tape_entry_index].gradient_tensor.data[0] = 1.0f;
                }
                size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                kl_divergence_gradients(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[prediction_tape_entry_index].gradient_tensor, &dag->tape_entries[target_tape_entry_index].gradient_tensor);
                break;
            }
            case OPERATION_TYPE_DROPOUT: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                StochasticRegularizationConfiguration* config = &dag->nodes[node_index].op_config.stochastic_regularization;
                uint64_t seed = config->base_seed ^ (dag->tape_entries[tape_entry_index].time_step * 0x9e3779b97f4a7c15ULL) ^ ((uint64_t)node_index * 0xbf58476d1ce4e5b9ULL);
                dropout_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor, config->probability, seed);
                break;
            }
            case OPERATION_TYPE_DROPPATH: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                StochasticRegularizationConfiguration* config = &dag->nodes[node_index].op_config.stochastic_regularization;
                uint64_t seed = config->base_seed ^ (dag->tape_entries[tape_entry_index].time_step * 0x9e3779b97f4a7c15ULL) ^ ((uint64_t)node_index * 0xbf58476d1ce4e5b9ULL);
                droppath_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor, config->probability, config->sample_dimension, seed);
                break;
            }
            case OPERATION_TYPE_ZONEOUT: {
                size_t current_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t previous_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                StochasticRegularizationConfiguration* config = &dag->nodes[node_index].op_config.stochastic_regularization;
                uint64_t seed = config->base_seed ^ (dag->tape_entries[tape_entry_index].time_step * 0x9e3779b97f4a7c15ULL) ^ ((uint64_t)node_index * 0xbf58476d1ce4e5b9ULL);
                zoneout_gradients(dag->tape_entries[current_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[current_tape_entry_index].gradient_tensor, &dag->tape_entries[previous_tape_entry_index].gradient_tensor, config->probability, seed);
                break;
            }
            default: { break; }
        }
    }
}

void stochastic_gradient_descent(Tensor* parameter_tensor, const Tensor gradient_tensor, StochasticGradientDescentConfiguration config) {
    for(size_t i = 0; i < parameter_tensor->total_elements; i++) {
        parameter_tensor->data[i] -= gradient_tensor.data[i] * config.learning_rate;
    }
}

void adam(Tensor* parameter_tensor, const Tensor gradient_tensor, AdamConfiguration* config) {
    if(config->first_moment.data == NULL) {
        config->first_moment = create_tensor(parameter_tensor->dims);
        set_tensor_data_to_zero(&config->first_moment);
    }
    if(config->second_moment.data == NULL) {
        config->second_moment = create_tensor(parameter_tensor->dims);
        set_tensor_data_to_zero(&config->second_moment);
    }

    config->time_step++;
    float first_moment_bias_correction = 1.0f - powf(config->beta1, (float)config->time_step);
    float second_moment_bias_correction = 1.0f - powf(config->beta2, (float)config->time_step);

    for(size_t i = 0; i < parameter_tensor->total_elements; i++) {
        float gradient = gradient_tensor.data[i];

        config->first_moment.data[i] = (config->beta1 * config->first_moment.data[i]) + ((1.0f - config->beta1) * gradient);
        config->second_moment.data[i] = (config->beta2 * config->second_moment.data[i]) + ((1.0f - config->beta2) * gradient * gradient);

        float corrected_first_moment = config->first_moment.data[i] / first_moment_bias_correction;
        float corrected_second_moment = config->second_moment.data[i] / second_moment_bias_correction;

        parameter_tensor->data[i] -= config->learning_rate * corrected_first_moment / (sqrtf(corrected_second_moment) + config->epsilon);
    }
}

void muon(Tensor* parameter_tensor, const Tensor gradient_tensor, MuonConfiguration* config) {
    const float NEWTON_SCHULZ_A = 3.4445f;
    const float NEWTON_SCHULZ_B = -4.7750f;
    const float NEWTON_SCHULZ_C = 2.0315f;
    const float FROBENIUS_NORM_EPSILON = 1e-7f;

    if(config->momentum_buffer.data == NULL) {
        config->momentum_buffer = create_tensor(parameter_tensor->dims);
        set_tensor_data_to_zero(&config->momentum_buffer);
    }

    const size_t ROW_DIMENSION = TENSOR_DIMS - 2;
    const size_t COLUMN_DIMENSION = TENSOR_DIMS - 1;
    size_t total_rows = parameter_tensor->dims[ROW_DIMENSION];
    size_t total_columns = parameter_tensor->dims[COLUMN_DIMENSION];
    size_t matrix_size = total_rows * total_columns;

    size_t total_slices = 1;
    for(size_t i = 0; i < ROW_DIMENSION; i++) {
        total_slices *= parameter_tensor->dims[i];
    }

    float* orthogonalized_update = malloc(matrix_size * sizeof(float));
    float* gram_matrix = malloc(total_rows * total_rows * sizeof(float));
    float* polynomial_matrix = malloc(total_rows * total_rows * sizeof(float));
    float* next_orthogonalized_update = malloc(matrix_size * sizeof(float));

    float update_scale = 1.0f;
    if(total_rows > total_columns) {
        update_scale = sqrtf((float)total_rows / (float)total_columns);
    }

    for(size_t slice_index = 0; slice_index < total_slices; slice_index++) {
        size_t slice_offset = slice_index * matrix_size;

        for(size_t i = 0; i < matrix_size; i++) {
            config->momentum_buffer.data[slice_offset + i] = (config->momentum * config->momentum_buffer.data[slice_offset + i]) + gradient_tensor.data[slice_offset + i];
        }

        for(size_t i = 0; i < matrix_size; i++) {
            if(config->nesterov) {
                orthogonalized_update[i] = gradient_tensor.data[slice_offset + i] + (config->momentum * config->momentum_buffer.data[slice_offset + i]);
            }
            else {
                orthogonalized_update[i] = config->momentum_buffer.data[slice_offset + i];
            }
        }

        double squared_frobenius_norm = 0.0;
        for(size_t i = 0; i < matrix_size; i++) {
            squared_frobenius_norm += (double)orthogonalized_update[i] * (double)orthogonalized_update[i];
        }
        float inverse_frobenius_norm = 1.0f / (sqrtf((float)squared_frobenius_norm) + FROBENIUS_NORM_EPSILON);
        for(size_t i = 0; i < matrix_size; i++) {
            orthogonalized_update[i] *= inverse_frobenius_norm;
        }

        for(size_t step = 0; step < config->newton_schulz_steps; step++) {
            for(size_t row_index = 0; row_index < total_rows; row_index++) {
                for(size_t row_index2 = 0; row_index2 < total_rows; row_index2++) {
                    float accumulator = 0.0f;
                    for(size_t column_index = 0; column_index < total_columns; column_index++) {
                        accumulator += orthogonalized_update[(row_index * total_columns) + column_index] * orthogonalized_update[(row_index2 * total_columns) + column_index];
                    }
                    gram_matrix[(row_index * total_rows) + row_index2] = accumulator;
                }
            }
            for(size_t row_index = 0; row_index < total_rows; row_index++) {
                for(size_t row_index2 = 0; row_index2 < total_rows; row_index2++) {
                    float gram_squared_value = 0.0f;
                    for(size_t k = 0; k < total_rows; k++) {
                        gram_squared_value += gram_matrix[(row_index * total_rows) + k] * gram_matrix[(k * total_rows) + row_index2];
                    }
                    polynomial_matrix[(row_index * total_rows) + row_index2] = (NEWTON_SCHULZ_B * gram_matrix[(row_index * total_rows) + row_index2]) + (NEWTON_SCHULZ_C * gram_squared_value);
                }
            }
            for(size_t row_index = 0; row_index < total_rows; row_index++) {
                for(size_t column_index = 0; column_index < total_columns; column_index++) {
                    float accumulator = 0.0f;
                    for(size_t k = 0; k < total_rows; k++) {
                        accumulator += polynomial_matrix[(row_index * total_rows) + k] * orthogonalized_update[(k * total_columns) + column_index];
                    }
                    next_orthogonalized_update[(row_index * total_columns) + column_index] = (NEWTON_SCHULZ_A * orthogonalized_update[(row_index * total_columns) + column_index]) + accumulator;
                }
            }
            for(size_t i = 0; i < matrix_size; i++) {
                orthogonalized_update[i] = next_orthogonalized_update[i];
            }
        }

        for(size_t i = 0; i < matrix_size; i++) {
            parameter_tensor->data[slice_offset + i] -= config->learning_rate * update_scale * orthogonalized_update[i];
        }
    }

    free(orthogonalized_update);
    free(gram_matrix);
    free(polynomial_matrix);
    free(next_orthogonalized_update);
}

void ano(Tensor* parameter_tensor, const Tensor gradient_tensor, AnoConfiguration* config) {
    if(config->first_moment.data == NULL) {
        config->first_moment = create_tensor(parameter_tensor->dims);
        set_tensor_data_to_zero(&config->first_moment);
    }
    if(config->second_moment.data == NULL) {
        config->second_moment = create_tensor(parameter_tensor->dims);
        set_tensor_data_to_zero(&config->second_moment);
    }

    for(size_t i = 0; i < parameter_tensor->total_elements; i++) {
        float gradient = gradient_tensor.data[i];
        float gradient_squared = gradient * gradient;

        config->first_moment.data[i] = (config->beta1 * config->first_moment.data[i]) + ((1.0f - config->beta1) * gradient);

        float variance_difference = config->second_moment.data[i] - gradient_squared;
        float variance_sign = (variance_difference > 0.0f) ? 1.0f : ((variance_difference < 0.0f) ? -1.0f : 0.0f);
        config->second_moment.data[i] -= (1.0f - config->beta2) * variance_sign * gradient_squared;

        float momentum_value = config->first_moment.data[i];
        float momentum_sign = (momentum_value > 0.0f) ? 1.0f : ((momentum_value < 0.0f) ? -1.0f : 0.0f);
        float gradient_magnitude = (gradient >= 0.0f) ? gradient : (-1.0f * gradient);

        float parameter_value = parameter_tensor->data[i];
        float scaled_step = (config->learning_rate / (sqrtf(config->second_moment.data[i]) + config->epsilon)) * gradient_magnitude * momentum_sign;
        parameter_tensor->data[i] = parameter_value - scaled_step - (config->learning_rate * config->weight_decay * parameter_value);
    }
}

void optimize_weights(DirectedAcyclicGraph* dag) {
    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        if(dag->nodes[node_index].op_type != OPERATION_TYPE_PARAMETERS) { continue; }
        if(dag->nodes[node_index].total_tape_entries == 0) { continue; }

        ParameterConfiguration* parameter_config = &dag->nodes[node_index].op_config.parameters;
        if(!parameter_config->allow_parameter_updates) { continue; }

        size_t tape_entry_index = dag->nodes[node_index].tape_entry_indices[0];
        Tensor* parameter_tensor = &dag->tape_entries[tape_entry_index].output_tensor;
        Tensor gradient_tensor = dag->tape_entries[tape_entry_index].gradient_tensor;
        if(parameter_tensor->data == NULL || gradient_tensor.data == NULL) { continue; }

        if(parameter_config->l2_strength != 0.0f) {
            for(size_t i = 0; i < gradient_tensor.total_elements; i++) {
                gradient_tensor.data[i] += parameter_config->l2_strength * parameter_tensor->data[i];
            }
        }
        if(parameter_config->l1_strength != 0.0f) {
            for(size_t i = 0; i < gradient_tensor.total_elements; i++) {
                float weight = parameter_tensor->data[i];
                float weight_sign = (weight > 0.0f) ? 1.0f : ((weight < 0.0f) ? -1.0f : 0.0f);
                gradient_tensor.data[i] += parameter_config->l1_strength * weight_sign;
            }
        }

        switch(parameter_config->optimizer.type) {
            case OPTIMIZER_TYPE_STOCHASTIC_GRADIENT_DESCENT: {
                stochastic_gradient_descent(parameter_tensor, gradient_tensor, parameter_config->optimizer.config.sgd);
                break;
            }
            case OPTIMIZER_TYPE_ADAM: {
                adam(parameter_tensor, gradient_tensor, &parameter_config->optimizer.config.adam);
                break;
            }
            case OPTIMIZER_TYPE_MUON: {
                muon(parameter_tensor, gradient_tensor, &parameter_config->optimizer.config.muon);
                break;
            }
            case OPTIMIZER_TYPE_ANO: {
                ano(parameter_tensor, gradient_tensor, &parameter_config->optimizer.config.ano);
                break;
            }
            default: { break; }
        }
    }
}
