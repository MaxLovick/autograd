#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint64_t state;
    uint64_t inc;
} PCGRandomNumberGenerator;

uint32_t pcg32_random_r(PCGRandomNumberGenerator* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

double pcg32_gaussian(PCGRandomNumberGenerator* rng) {
    double u1 = ldexp(pcg32_random_r(rng), -32);
    double u2 = ldexp(pcg32_random_r(rng), -32);

    if (u1 <= 0.0) u1 = 1e-10;

    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* Tensor Ops */

#define TENSOR_DIMENSIONS 5

typedef struct {
    size_t total_elements;
    size_t* dims;
    float* data;
} Tensor;

Tensor create_tensor(const size_t* dims) {
    Tensor output;
    if(dims) {
        output.total_elements = 1;
        output.dims = malloc(TENSOR_DIMENSIONS * sizeof(size_t));
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            output.dims[i] = dims[i];
            output.total_elements *= output.dims[i];
        }
        output.data = malloc(output.total_elements * sizeof(float));
    }
    else {
        output.total_elements = 0;
        output.dims = NULL;
        output.data = NULL;
    }
    return output;
}

Tensor deep_copy_tensor(const Tensor input) {
    Tensor output = create_tensor(input.dims);
    memcpy(output.data, input.data, input.total_elements * sizeof(float));
    return output;
}

void free_tensor(Tensor* tensor) {
    if(tensor) {
        tensor->total_elements = 0;
        free(tensor->dims);
        tensor->dims = NULL;
        free(tensor->data);
        tensor->data = NULL;
    }
}

void set_tensor_data_to_zeros(Tensor* tensor) {
    for(size_t i = 0; i < tensor->total_elements; i++) {
        tensor->data[i] = 0.0f;
    }
}

void set_tensor_data_to_ones(Tensor* tensor) {
    for(size_t i = 0; i < tensor->total_elements; i++) {
        tensor->data[i] = 1.0f;
    }
}

void add_gaussian_noise_to_tensor_data(Tensor* tensor, float mean, float std_dev, PCGRandomNumberGenerator* rng) {
    for(size_t i = 0; i < tensor->total_elements; i++ ) {
        tensor->data[i] += (mean + (std_dev * pcg32_gaussian(rng)));
    }
}

void print_tensor(const Tensor tensor) {
    size_t edge_items = 3;
    size_t summarize_threshold = 1000;
    int decimal_places = 4;

    size_t first_dim = TENSOR_DIMENSIONS - 1;
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        if(tensor.dims[i] != 1) {
            first_dim = i; break;
        }
    }
    size_t ndims = TENSOR_DIMENSIONS - first_dim;

    size_t dims[TENSOR_DIMENSIONS];
    size_t strides[TENSOR_DIMENSIONS];
    for(size_t d = 0; d < ndims; d++) {
        dims[d] = tensor.dims[first_dim + d];
    }
    strides[ndims - 1] = 1;
    for(size_t i = ndims - 1; i-- > 0;) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    bool summarize = (tensor.total_elements > summarize_threshold);
    bool dim_summarized[TENSOR_DIMENSIONS];
    for(size_t d = 0; d < ndims; d++) {
        dim_summarized[d] = (summarize && (dims[d] > 2 * edge_items));
    }

    float max_abs = 0.0f;
    for(size_t i = 0; i < tensor.total_elements; i++) {
        float a = fabsf(tensor.data[i]);
        if(a > max_abs) {
            max_abs = a;
        }
    }
    int integer_digits = (max_abs < 1.0f) ? 1 : (int)floorf(log10f(max_abs)) + 1;
    int field_width = 1 + integer_digits + 1 + decimal_places;

    printf("shape=[");
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        printf("%zu", tensor.dims[i]);
        if(i < TENSOR_DIMENSIONS - 1) {
            printf(", ");
        }
    }
    printf("])\n");

    printf("tensor(");
    size_t indent_base = 7;
    size_t indices[TENSOR_DIMENSIONS];
    size_t prev_indices[TENSOR_DIMENSIONS];
    for(size_t d = 0; d < ndims; d++) {
        indices[d] = 0;
    }

    bool first_element = true;
    bool done = false;

    while(!done) {
        size_t flat_idx = 0;
        for(size_t d = 0; d < ndims; d++) {
            flat_idx += indices[d] * strides[d];
        }

        if(first_element) {
            for(size_t d = 0; d < ndims; d++) {
                printf("[");
            }
            printf("%*.*f", field_width, decimal_places, tensor.data[flat_idx]);
            for(size_t d = 0; d < ndims; d++) {
                prev_indices[d] = indices[d];
            }
            first_element = false;
        }
        else {
            size_t changed_dim = ndims - 1;
            for(size_t d = 0; d < ndims; d++) {
                if(indices[d] != prev_indices[d]) {
                    changed_dim = d; break;
                }
            }
            bool has_gap = (dim_summarized[changed_dim] && (prev_indices[changed_dim] + 1 != indices[changed_dim]));

            if(changed_dim == ndims - 1) {
                if(has_gap) {
                    printf(", ...");
                }
                printf(", %*.*f", field_width, decimal_places, tensor.data[flat_idx]);
            }
            else {
                size_t brackets = ndims - 1 - changed_dim;
                for(size_t i = 0; i < brackets; i++) {
                    printf("]");
                }
                printf(",");

                size_t blank_lines = (brackets > 1) ? brackets - 1 : 0;
                if(has_gap) {
                    for(size_t b = 0; b < blank_lines; b++) {
                        printf("\n");
                    }
                    printf("\n");
                    for(size_t s = 0; s < indent_base + changed_dim + 1; s++) {
                        printf(" ");
                    }
                    printf("...,");
                }

                for(size_t b = 0; b < blank_lines; b++) {
                    printf("\n");
                }
                printf("\n");
                for(size_t s = 0; s < indent_base + changed_dim + 1; s++) {
                    printf(" ");
                }

                for(size_t i = 0; i < brackets; i++) {
                    printf("[");
                }
                printf("%*.*f", field_width, decimal_places, tensor.data[flat_idx]);
            }

            for(size_t d = 0; d < ndims; d++) {
                prev_indices[d] = indices[d];
            }
        }

        bool carry = true;
        for(size_t d = ndims; d-- > 0 && carry;) {
            indices[d]++;
            if(dim_summarized[d] && indices[d] == edge_items) {
                indices[d] = dims[d] - edge_items;
                carry = false;
            }
            else if(indices[d] >= dims[d]) {
                indices[d] = 0;
            }
            else {
                carry = false;
            }
        }
        if(carry) {
            done = true;
        }
    }

    for(size_t d = 0; d < ndims; d++) {
        printf("]");
    }
}

Tensor resize(const Tensor A, const size_t* new_dims) {
    Tensor output = create_tensor(new_dims);
    for(size_t i = 0; i < A.total_elements; i++) {
        output.data[i] = A.data[i];
    }
    return output;
}

void resize_gradients(const Tensor A, const Tensor upstream_gradients, Tensor* gradA) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        gradA->data[i] += upstream_gradients.data[i];
    }
}

Tensor concat(const Tensor A, const Tensor B, const size_t concat_dim) {
    size_t* output_dims = malloc(TENSOR_DIMENSIONS * sizeof(size_t));
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_dims[i] = (i == concat_dim) ? (A.dims[i] + B.dims[i]) : A.dims[i];
    }
    Tensor output = create_tensor(output_dims);

    if(concat_dim == 0) {
        memcpy(output.data, A.data, A.total_elements * sizeof(float));
        memcpy(output.data + A.total_elements, B.data, B.total_elements * sizeof(float));
    }
    else {
        size_t inner_block_size = 1;
        for(size_t i = concat_dim + 1; i < TENSOR_DIMENSIONS; i++) {
            inner_block_size *= A.dims[i];
        }
        size_t total_inner_blocks = 1;
        for(size_t i = 0; i < concat_dim; i++) {
            total_inner_blocks *= A.dims[i];
        }
        size_t block_sizeA = A.dims[concat_dim] * inner_block_size;
        size_t block_sizeB = B.dims[concat_dim] * inner_block_size;
        size_t output_block_size = output.dims[concat_dim] * inner_block_size;

        for(size_t inner_block = 0; inner_block < total_inner_blocks; inner_block++) {
            for(size_t i = 0; i < block_sizeA; i++) {
                output.data[(inner_block * output_block_size) + i] = A.data[(inner_block * block_sizeA) + i];
            }
            for(size_t i = 0; i < block_sizeB; i++) {
                output.data[(inner_block * output_block_size) + block_sizeA + i] = B.data[(inner_block * block_sizeB) + i];
            }
        }
    }
    return output;
}

void concat_gradients(const Tensor A, const Tensor B, const Tensor upstream_gradients, Tensor* gradA, Tensor* gradB, size_t concat_dim) {
    if(concat_dim == 0) {
        for(size_t i = 0; i < A.total_elements; i++) {
            gradA->data[i] += upstream_gradients.data[i];
        }
        for(size_t i = 0; i < B.total_elements; i++) {
            gradB->data[i] += upstream_gradients.data[A.total_elements + i];
        }
    }
    else {
        size_t inner_block_size = 1;
        for(size_t i = concat_dim + 1; i < TENSOR_DIMENSIONS; i++) {
            inner_block_size *= A.dims[i];
        }
        size_t total_inner_blocks = 1;
        for(size_t i = 0; i < concat_dim; i++) {
            total_inner_blocks *= A.dims[i];
        }
        size_t block_sizeA = A.dims[concat_dim] * inner_block_size;
        size_t block_sizeB = B.dims[concat_dim] * inner_block_size;
        size_t upstream_block_size = upstream_gradients.dims[concat_dim] * inner_block_size;

        for(size_t inner_block = 0; inner_block < total_inner_blocks; inner_block++) {
            for(size_t i = 0; i < block_sizeA; i++) {
                gradA->data[(inner_block * block_sizeA) + i] += upstream_gradients.data[(inner_block * upstream_block_size) + i];
            }
            for(size_t i = 0; i < block_sizeB; i++) {
                gradB->data[(inner_block * block_sizeB) + i] += upstream_gradients.data[(inner_block * upstream_block_size) + block_sizeA + i];
            }
        }
    }
}

Tensor add(const Tensor A, const Tensor B) {
    size_t output_dims[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        if(A.dims[i] == B.dims[i]) { output_dims[i] = A.dims[i]; }
        else if(A.dims[i] > 1 && B.dims[i] == 1) { output_dims[i] = A.dims[i]; }
        else if(A.dims[i] == 1 && B.dims[i] > 1) { output_dims[i] = B.dims[i]; }
    }
    Tensor output = create_tensor(output_dims);

    size_t output_indices[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_indices[i] = 0;
    }

    size_t stridesA[TENSOR_DIMENSIONS];
    size_t stridesB[TENSOR_DIMENSIONS];
    stridesA[TENSOR_DIMENSIONS - 1] = 1;
    stridesB[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        stridesA[i] = stridesA[i + 1] * A.dims[i + 1];
        stridesB[i] = stridesB[i + 1] * B.dims[i + 1];
    }

    for(size_t output_index = 0; output_index < output.total_elements; output_index++) {
        size_t indexA = 0;
        size_t indexB = 0;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            if(A.dims[i] > 1) { indexA += output_indices[i] * stridesA[i]; }
            if(B.dims[i] > 1) { indexB += output_indices[i] * stridesB[i]; }
        }
        output.data[output_index] = A.data[indexA] + B.data[indexB];

        for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] >= output.dims[i]) { output_indices[i] = 0; }
            else { break; }
        }
    }
    return output;
}

void add_gradients(const Tensor A, const Tensor B, const Tensor upstream_gradients, Tensor* gradA, Tensor* gradB) {
    size_t output_indices[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_indices[i] = 0;
    }

    size_t stridesA[TENSOR_DIMENSIONS];
    size_t stridesB[TENSOR_DIMENSIONS];
    stridesA[TENSOR_DIMENSIONS - 1] = 1;
    stridesB[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        stridesA[i] = stridesA[i + 1] * A.dims[i + 1];
        stridesB[i] = stridesB[i + 1] * B.dims[i + 1];
    }

    for(size_t output_index = 0; output_index < upstream_gradients.total_elements; output_index++) {
        size_t indexA = 0;
        size_t indexB = 0;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            if(A.dims[i] > 1) { indexA += output_indices[i] * stridesA[i]; }
            if(B.dims[i] > 1) { indexB += output_indices[i] * stridesB[i]; }
        }
        gradA->data[indexA] += upstream_gradients.data[output_index];
        gradB->data[indexB] += upstream_gradients.data[output_index];

        for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] >= upstream_gradients.dims[i]) { output_indices[i] = 0; }
            else { break; }
        }
    }
}

Tensor scalar_tensor_mul(const Tensor scalar, const Tensor tensor) {
    Tensor output = create_tensor(tensor.dims);
    for(size_t i = 0; i < tensor.total_elements; i++) {
        output.data[i] = scalar.data[0] * tensor.data[i];
    }
    return output;
}

void scalar_tensor_mul_gradients(const Tensor scalar, const Tensor tensor, const Tensor upstream_gradients, Tensor* scalar_grad, Tensor* tensor_grad) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        scalar_grad->data[0] += tensor.data[i] * upstream_gradients.data[i];
        tensor_grad->data[i] += scalar.data[0] * upstream_gradients.data[i];
    }
}

Tensor elementwise_mul(const Tensor A, const Tensor B) {
    Tensor output = create_tensor(A.dims);
    for(size_t i = 0; i < A.total_elements; i++) {
        output.data[i] = A.data[i] * B.data[i];
    }
    return output;
}

void elementwise_mul_gradients(const Tensor A, const Tensor B, const Tensor upstream_gradients, Tensor* gradA, Tensor* gradB) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        gradA->data[i] += B.data[i] * upstream_gradients.data[i];
        gradB->data[i] += A.data[i] * upstream_gradients.data[i];
    }
}

Tensor matmul(const Tensor A, const Tensor B) {
    size_t output_dims[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < (TENSOR_DIMENSIONS - 2); i++) {
        if(A.dims[i] == B.dims[i]) { output_dims[i] = A.dims[i]; }
        else if(A.dims[i] > 1 && B.dims[i] == 1) { output_dims[i] = A.dims[i]; }
        else if(A.dims[i] == 1 && B.dims[i] > 1) { output_dims[i] = B.dims[i]; }
    }
    output_dims[TENSOR_DIMENSIONS - 2] = A.dims[TENSOR_DIMENSIONS - 2];
    output_dims[TENSOR_DIMENSIONS - 1] = B.dims[TENSOR_DIMENSIONS - 1];
    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zeros(&output);
    size_t common_dim = A.dims[TENSOR_DIMENSIONS - 1];

    size_t output_indices[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_indices[i] = 0;
    }

    size_t stridesA[TENSOR_DIMENSIONS];
    size_t stridesB[TENSOR_DIMENSIONS];
    stridesA[TENSOR_DIMENSIONS - 1] = 1;
    stridesB[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        stridesA[i] = stridesA[i + 1] * A.dims[i + 1];
        stridesB[i] = stridesB[i + 1] * B.dims[i + 1];
    }

    for(size_t output_index = 0; output_index < output.total_elements; output_index++) {
        size_t indexA_base = 0;
        size_t indexB_base = 0;
        for(size_t i = 0; i < (TENSOR_DIMENSIONS - 2); i++) {
            if(A.dims[i] > 1) { indexA_base += output_indices[i] * stridesA[i]; }
            if(B.dims[i] > 1) { indexB_base += output_indices[i] * stridesB[i]; }
        }

        for(size_t i = 0; i < common_dim; i++) {
            size_t indexA = indexA_base + (output_indices[TENSOR_DIMENSIONS - 2] * stridesA[TENSOR_DIMENSIONS - 2]) + (i * stridesA[TENSOR_DIMENSIONS - 1]);
            size_t indexB = indexB_base + (output_indices[TENSOR_DIMENSIONS - 1] * stridesB[TENSOR_DIMENSIONS - 1]) + (i * stridesB[TENSOR_DIMENSIONS - 2]);
            output.data[output_index] += A.data[indexA] * B.data[indexB];
        }

        for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] >= output.dims[i]) { output_indices[i] = 0; }
            else { break; }
        }
    }
    return output;
}

void matmul_gradients(const Tensor A, const Tensor B, const Tensor upstream_gradients, Tensor* gradA, Tensor* gradB) {
    size_t common_dim = A.dims[TENSOR_DIMENSIONS - 1];
    size_t output_indices[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_indices[i] = 0;
    }

    size_t stridesA[TENSOR_DIMENSIONS];
    size_t stridesB[TENSOR_DIMENSIONS];
    size_t upstream_gradients_strides[TENSOR_DIMENSIONS];
    stridesA[TENSOR_DIMENSIONS - 1] = 1;
    stridesB[TENSOR_DIMENSIONS - 1] = 1;
    upstream_gradients_strides[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        stridesA[i] = stridesA[i + 1] * A.dims[i + 1];
        stridesB[i] = stridesB[i + 1] * B.dims[i + 1];
        upstream_gradients_strides[i] = upstream_gradients_strides[i + 1] * upstream_gradients.dims[i + 1];
    }

    for(size_t output_index = 0; output_index < upstream_gradients.total_elements; output_index++) {
        size_t indexA_base = 0;
        size_t indexB_base = 0;
        for(size_t i = 0; i < (TENSOR_DIMENSIONS - 2); i++) {
            if(A.dims[i] > 1) { indexA_base += output_indices[i] * stridesA[i]; }
            if(B.dims[i] > 1) { indexB_base += output_indices[i] * stridesB[i]; }
        }

        size_t upstream_gradient_index = 0;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            upstream_gradient_index += output_indices[i] * upstream_gradients_strides[i];
        }
        float upstream_gradient = upstream_gradients.data[upstream_gradient_index];

        for(size_t i = 0; i < common_dim; i++) {
            size_t indexA = indexA_base + (output_indices[TENSOR_DIMENSIONS - 2] * stridesA[TENSOR_DIMENSIONS - 2]) + (i * stridesA[TENSOR_DIMENSIONS - 1]);
            size_t indexB = indexB_base + (output_indices[TENSOR_DIMENSIONS - 1] * stridesB[TENSOR_DIMENSIONS - 1]) + (i * stridesB[TENSOR_DIMENSIONS - 2]);

            gradA->data[indexA] += B.data[indexB] * upstream_gradient;
            gradB->data[indexB] += A.data[indexA] * upstream_gradient;
        }

        for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
            output_indices[i]++;
            if(output_indices[i] >= upstream_gradients.dims[i]) { output_indices[i] = 0; }
            else { break; }
        }
    }
}

Tensor convolution(const Tensor A, const Tensor B) {
    size_t output_dims[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_dims[i] = (A.dims[i] - B.dims[i]) + 1;
    }
    Tensor output = create_tensor(output_dims);
    set_tensor_data_to_zeros(&output);

    size_t stridesA[TENSOR_DIMENSIONS];
    size_t stridesB[TENSOR_DIMENSIONS];
    stridesA[TENSOR_DIMENSIONS - 1] = 1;
    stridesB[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        stridesA[i] = stridesA[i + 1] * A.dims[i + 1];
        stridesB[i] = stridesB[i + 1] * B.dims[i + 1];
    }

    size_t indicesA_base[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        indicesA_base[i] = 0;
    }

    size_t indicesA[TENSOR_DIMENSIONS];
    size_t indicesB[TENSOR_DIMENSIONS];
    for(size_t output_index = 0; output_index < output.total_elements; output_index++) {
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            indicesA[i] = indicesA_base[i];
            indicesB[i] = 0;
        }

        for(size_t element_in_B = 0; element_in_B < B.total_elements; element_in_B++) {
            size_t indexA = 0;
            size_t indexB = 0;
            for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
                indexA += indicesA[i] * stridesA[i];
                indexB += indicesB[i] * stridesB[i];
            }

            output.data[output_index] += A.data[indexA] * B.data[indexB];

            for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
                indicesB[i]++;
                if(indicesB[i] >= B.dims[i]) { indicesB[i] = 0; }
                else { break; }
            }
            for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
                indicesA[i] = indicesA_base[i] + indicesB[i];
            }
        }

        for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
            indicesA_base[i]++;
            if(indicesA_base[i] >= A.dims[i]) { indicesA_base[i] = 0; }
            else { break; }
        }
    }
    return output;
}

void convolution_gradients(const Tensor A, const Tensor B, const Tensor upstream_gradients, Tensor* gradA, Tensor* gradB) {
    size_t stridesA[TENSOR_DIMENSIONS];
    size_t stridesB[TENSOR_DIMENSIONS];
    size_t upstream_strides[TENSOR_DIMENSIONS];
    stridesA[TENSOR_DIMENSIONS - 1] = 1;
    stridesB[TENSOR_DIMENSIONS - 1] = 1;
    upstream_strides[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        stridesA[i] = stridesA[i + 1] * A.dims[i + 1];
        stridesB[i] = stridesB[i + 1] * B.dims[i + 1];
        upstream_strides[i] = upstream_strides[i + 1] * upstream_gradients.dims[i + 1];
    }

    size_t indicesA_base[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        indicesA_base[i] = 0;
    }

    size_t indicesA[TENSOR_DIMENSIONS];
    size_t indicesB[TENSOR_DIMENSIONS];
    for(size_t output_index = 0; output_index < upstream_gradients.total_elements; output_index++) {
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            indicesA[i] = indicesA_base[i];
            indicesB[i] = 0;
        }

        for(size_t element_in_B = 0; element_in_B < B.total_elements; element_in_B++) {
            size_t indexA = 0;
            size_t indexB = 0;
            for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
                indexA += indicesA[i] * stridesA[i];
                indexB += indicesB[i] * stridesB[i];
            }

            gradA->data[indexA] += B.data[indexB] * upstream_gradients.data[output_index];
            gradB->data[indexB] += A.data[indexA] * upstream_gradients.data[output_index];

            for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
                indicesB[i]++;
                if(indicesB[i] >= B.dims[i]) { indicesB[i] = 0; }
                else { break; }
            }
            for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
                indicesA[i] = indicesA_base[i] + indicesB[i];
            }
        }

        for(size_t i = TENSOR_DIMENSIONS; i-- > 0;) {
            indicesA_base[i]++;
            if(indicesA_base[i] >= A.dims[i]) { indicesA_base[i] = 0; }
            else { break; }
        }
    }
}


Tensor layer_norm(const Tensor A, const Tensor gamma, const Tensor beta, float epsilon, bool uses_batches, size_t batch_dim) {
    Tensor output = create_tensor(A.dims);

    size_t total_batches;
    size_t total_elements_inside_batch = 1;
    if(uses_batches) {
        total_batches = A.dims[batch_dim];
        for(size_t i = (batch_dim + 1); i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }
    else {
        batch_dim = 0;
        total_batches = 1;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }

    size_t strides[TENSOR_DIMENSIONS];
    strides[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        strides[i] = strides[i + 1] * A.dims[i + 1];
    }

    for(size_t batch_index = 0; batch_index < total_batches; batch_index++) {
        size_t index_base = batch_index * strides[batch_dim];
        double mean = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            mean += (double)A.data[index_base + i];
        }
        mean /= (double)total_elements_inside_batch;

        double squared_difference = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double mean_minus_value = (double)A.data[index_base + i] - mean;
            squared_difference += mean_minus_value * mean_minus_value;
        }
        double variance = squared_difference / (double)total_elements_inside_batch;
        double std_dev = sqrt(variance + epsilon);

        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            output.data[index_base + i] = (A.data[index_base + i] - (float)mean) / (float)std_dev;
            output.data[index_base + i] *= gamma.data[i];
            output.data[index_base + i] += beta.data[i];
        }
    }

    return output;
}

void layer_norm_gradients(const Tensor A, const Tensor gamma, const Tensor beta, float epsilon, const Tensor upstream_gradients, Tensor* gradA, Tensor* grad_gamma, Tensor* grad_beta, bool uses_batches, size_t batch_dim) {
    size_t total_batches;
    size_t total_elements_inside_batch = 1;
    if(uses_batches) {
        total_batches = A.dims[batch_dim];
        for(size_t i = (batch_dim + 1); i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }
    else {
        batch_dim = 0;
        total_batches = 1;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }

    size_t strides[TENSOR_DIMENSIONS];
    strides[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        strides[i] = strides[i + 1] * A.dims[i + 1];
    }

    for(size_t batch_index = 0; batch_index < total_batches; batch_index++) {
        size_t index_base = batch_index * strides[batch_dim];
        double mean = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            mean += (double)A.data[index_base + i];
        }
        mean /= (double)total_elements_inside_batch;

        double squared_difference = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double mean_minus_value = (double)A.data[index_base + i] - mean;
            squared_difference += mean_minus_value * mean_minus_value;
        }
        double variance = squared_difference / (double)total_elements_inside_batch;
        double inverse_std_dev = 1.0 / sqrt(variance + epsilon);

        double normalized_value_gradient_sum = 0.0;
        double normalized_gradient_dot_normalized_sum = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double normalized_value = ((double)A.data[index_base + i] - mean) * inverse_std_dev;
            double upstream_grad = (double)upstream_gradients.data[index_base + i];
            double normalized_value_gradient = upstream_grad * (double)gamma.data[i];
            grad_gamma->data[i] += (float)(normalized_value * upstream_grad);
            grad_beta->data[i] += (float)upstream_grad;

            normalized_value_gradient_sum += normalized_value_gradient;
            normalized_gradient_dot_normalized_sum += normalized_value_gradient * normalized_value;
        }

        double scale = inverse_std_dev / (double)total_elements_inside_batch;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double normalized_value = ((double)A.data[index_base + i] - mean) * inverse_std_dev;
            double upstream_grad = (double)upstream_gradients.data[index_base + i];
            double normalized_value_gradient = upstream_grad * (double)gamma.data[i];
            double term = ((double)total_elements_inside_batch * normalized_value_gradient) - normalized_value_gradient_sum - (normalized_value * normalized_gradient_dot_normalized_sum);
            gradA->data[index_base + i] += (float)(scale * term);
        }
    }
}

Tensor rms_norm(const Tensor A, const Tensor gamma, float epsilon, bool uses_batches, size_t batch_dim) {
    Tensor output = create_tensor(A.dims);

    size_t total_batches;
    size_t total_elements_inside_batch = 1;
    if(uses_batches) {
        total_batches = A.dims[batch_dim];
        for(size_t i = (batch_dim + 1); i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }
    else {
        batch_dim = 0;
        total_batches = 1;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }

    size_t strides[TENSOR_DIMENSIONS];
    strides[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        strides[i] = strides[i + 1] * A.dims[i + 1];
    }

    for(size_t batch_index = 0; batch_index < total_batches; batch_index++) {
        size_t index_base = batch_index * strides[batch_dim];

        double sum_of_squares = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double value = (double)A.data[index_base + i];
            sum_of_squares += value * value;
        }
        double rms = sqrt(sum_of_squares / (double)total_elements_inside_batch + epsilon);

        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            output.data[index_base + i] = (A.data[index_base + i] / (float)rms) * gamma.data[i];
        }
    }

    return output;
}

void rms_norm_gradients(const Tensor A, const Tensor gamma, float epsilon, const Tensor upstream_gradients, Tensor* gradA, Tensor* grad_gamma, bool uses_batches, size_t batch_dim) {
    size_t total_batches;
    size_t total_elements_inside_batch = 1;
    if(uses_batches) {
        total_batches = A.dims[batch_dim];
        for(size_t i = (batch_dim + 1); i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }
    else {
        batch_dim = 0;
        total_batches = 1;
        for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            total_elements_inside_batch *= A.dims[i];
        }
    }

    size_t strides[TENSOR_DIMENSIONS];
    strides[TENSOR_DIMENSIONS - 1] = 1;
    for(size_t i = (TENSOR_DIMENSIONS - 1); i-- > 0;) {
        strides[i] = strides[i + 1] * A.dims[i + 1];
    }

    for(size_t batch_index = 0; batch_index < total_batches; batch_index++) {
        size_t index_base = batch_index * strides[batch_dim];

        double sum_of_squares = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double value = (double)A.data[index_base + i];
            sum_of_squares += value * value;
        }
        double mean_of_squares = sum_of_squares / (double)total_elements_inside_batch;
        double rms = sqrt(mean_of_squares + epsilon);
        double inverse_rms = 1.0 / rms;

        double scaled_gradient_dot_input_sum = 0.0;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double upstream_grad = (double)upstream_gradients.data[index_base + i];
            double scaled_gradient = upstream_grad * (double)gamma.data[i];
            double normalized_value = (double)A.data[index_base + i] * inverse_rms;

            grad_gamma->data[i] += (float)(normalized_value * upstream_grad);
            scaled_gradient_dot_input_sum += scaled_gradient * (double)A.data[index_base + i];
        }

        double scale = inverse_rms / (double)total_elements_inside_batch;
        for(size_t i = 0; i < total_elements_inside_batch; i++) {
            double upstream_grad = (double)upstream_gradients.data[index_base + i];
            double scaled_gradient = upstream_grad * (double)gamma.data[i];
            double term = ((double)total_elements_inside_batch * scaled_gradient) - ((double)A.data[index_base + i] * inverse_rms * inverse_rms * scaled_gradient_dot_input_sum);
            gradA->data[index_base + i] += (float)(scale * term);
        }
    }
}

Tensor sigmoid(const Tensor A) {
    Tensor output = create_tensor(A.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = A.data[i];
        if(x >= 0.0f) {
            output.data[i] = 1.0f / (1.0f + expf((-1.0f * A.data[i])));
        } else {
            float z = expf(x);
            output.data[i] = z / (1.0f + z);
        }
    }
    return output;
}

void sigmoid_gradients(const Tensor A, const Tensor upstream_gradients, Tensor* gradA) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        gradA->data[i] += upstream_gradients.data[i] * A.data[i] * (1.0f - A.data[i]);
    }
}

Tensor softmax(const Tensor A, size_t dim) {
    size_t output_dims[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) { output_dims[i] = A.dims[i]; }
    Tensor output = create_tensor(A.dims);

    size_t aggregated_dim_size = A.dims[dim];

    size_t outer_count = 1;
    for(size_t d = 0; d < dim; d++) { outer_count *= A.dims[d]; }

    size_t inner_count = 1;
    for(size_t d = dim + 1; d < TENSOR_DIMENSIONS; d++) { inner_count *= A.dims[d]; }

    for(size_t outer_index = 0; outer_index < outer_count; outer_index++) {
        for(size_t inner_index = 0; inner_index < inner_count; inner_index++) {
            size_t base = outer_index * (aggregated_dim_size * inner_count) + inner_index;

            float max_value = A.data[base + (0 * inner_count)];
            for(size_t i = 1; i < aggregated_dim_size; i++) {
                float x = A.data[base + (i * inner_count)];
                if(x > max_value) { max_value = x; }
            }

            float sum = 0.0f;
            for(size_t i = 0; i < aggregated_dim_size; i++) {
                float e = expf(A.data[base + i * inner_count] - max_value);
                output.data[base + i * inner_count] = e;
                sum += e;
            }

            float inverse = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            for(size_t i = 0; i < aggregated_dim_size; i++) {
                output.data[base + i * inner_count] *= inverse;
            }
        }
    }

    return output;
}

void softmax_gradients(const Tensor A, const Tensor upstream_gradients, Tensor* gradA, size_t dim) {
    size_t aggregated_dim_size = A.dims[dim];

    size_t outer_count = 1;
    for(size_t d = 0; d < dim; d++) { outer_count *= A.dims[d]; }

    size_t inner_count = 1;
    for(size_t d = dim + 1; d < TENSOR_DIMENSIONS; d++) { inner_count *= A.dims[d]; }

    for(size_t outer_index = 0; outer_index < outer_count; outer_index++) {
        for(size_t inner_index = 0; inner_index < inner_count; inner_index++) {
            size_t base = outer_index * (aggregated_dim_size * inner_count) + inner_index;

            float dot = 0.0f;
            for(size_t i = 0; i < aggregated_dim_size; i++) {
                dot += upstream_gradients.data[base + i * inner_count] * A.data[base + i * inner_count];
            }

            for(size_t i = 0; i < aggregated_dim_size; i++) {
                gradA->data[base + i * inner_count] += A.data[base + i * inner_count] * (upstream_gradients.data[base + i * inner_count] - dot);
            }
        }
    }
}

Tensor gelu(const Tensor A) {
    Tensor output = create_tensor(A.dims);
    const float INV_SQRT2 = 0.7071067811865475f;
    for(size_t i = 0; i < output.total_elements; i++) {
        float x = A.data[i];
        float y = 0.5f * x * (1.0f + erff(x * INV_SQRT2));
        output.data[i] = y;
    }
    return output;
}

void gelu_gradients(const Tensor A, const Tensor upstream_gradients, Tensor* gradA) {
    const float INV_SQRT2 = 0.7071067811865475f;
    const float INV_SQRT2_PI = 0.3989422804014327f;

    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float x = A.data[i];
        float cdf = 0.5f * (1.0f + erff(x * INV_SQRT2));
        float pdf = INV_SQRT2_PI * expf(-0.5f * x * x);
        float dx = cdf + x * pdf;
        gradA->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor swish(const Tensor A) {
    Tensor output = create_tensor(A.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float s;
        if(A.data[i] >= 0.0f) {
            float z = expf((-1.0f * A.data[i]));
            s = 1.0f / (1.0f + z);
        } else {
            float z = expf(A.data[i]);
            s = z / (1.0f + z);
        }
        output.data[i] = A.data[i] * s;
    }
    return output;
}

void swish_gradients(const Tensor A, const Tensor upstream_gradients, Tensor* gradA) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float s;
        if(A.data[i] >= 0.0f) {
            float z = expf(-1.0f * A.data[i]);
            s = 1.0f / (1.0f + z);
        } else {
            float z = expf(A.data[i]);
            s = z / (1.0f + z);
        }

        float dx = s + A.data[i] * s * (1.0f - s);
        gradA->data[i] += upstream_gradients.data[i] * dx;
    }
}


Tensor mish(const Tensor A) {
    Tensor output = create_tensor(A.dims);
    for(size_t i = 0; i < output.total_elements; i++) {
        float sp;
        if(A.data[i] > 20.0f) { sp = A.data[i]; }
        else if(A.data[i] < -20.0f) { sp = expf(A.data[i]); }
        else { sp = log1pf(expf(A.data[i])); }
        float tan = tanhf(sp);
        output.data[i] = A.data[i] * tan;
    }
    return output;
}

void mish_gradients(const Tensor A, const Tensor upstream_gradients, Tensor* gradA) {
    for(size_t i = 0; i < upstream_gradients.total_elements; i++) {
        float sp;
        if(A.data[i] > 20.0f) { sp = A.data[i]; }
        else if(A.data[i] < -20.0f) { sp = expf(A.data[i]); }
        else { sp = log1pf(expf(A.data[i])); }

        float tan = tanhf(sp);
        float s;
        if(A.data[i] >= 0.0f) {
            float z = expf((-1.0f * A.data[i]));
            s = 1.0f / (1.0f + z);
        } else {
            float z = expf(A.data[i]);
            s = z / (1.0f + z);
        }
        float sech2 = 1.0f - tan * tan;

        float dx = tan + A.data[i] * s * sech2;
        gradA->data[i] += upstream_gradients.data[i] * dx;
    }
}

Tensor categorical_cross_entropy(const Tensor prediction, const Tensor target, float label_smoothing, size_t batch_dim, bool no_reduction, bool sum_reduction, bool mean_reduction) {
    size_t num_classes = prediction.dims[TENSOR_DIMENSIONS - 1];
    size_t num_batches = prediction.dims[batch_dim];
    float smoothing_per_class = label_smoothing / (float)num_classes;
    float target_scale = 1.0f - label_smoothing;
    float* sample_losses = malloc(num_batches * sizeof(float));
    for(size_t i = 0; i < num_batches; i++) {
        sample_losses[i] = 0.0f;
    }

    for(size_t batch_index = 0; batch_index < num_batches; batch_index++) {
        size_t outer_offset = batch_index * num_classes;
        float max_prediction_value = prediction.data[outer_offset];
        for(size_t i = 1; i < num_classes; i++) {
            if(max_prediction_value < prediction.data[outer_offset + i]) {
                max_prediction_value = prediction.data[outer_offset + i];
            }
        }

        float exp_sum = 0.0f;
        for(size_t i = 0; i < num_classes; i++) {
            exp_sum += expf(prediction.data[outer_offset + i] - max_prediction_value);
        }
        float log_exp_sum = logf(exp_sum);

        float sample_loss = 0.0f;
        for(size_t i = 0; i < num_classes; i++) {
            float prediction_probability = (prediction.data[outer_offset + i] - max_prediction_value) - log_exp_sum;
            float target_probability = target.data[outer_offset + i];
            target_probability = (target_probability * target_scale) + smoothing_per_class;
            sample_loss += (-1.0f * target_probability) * prediction_probability;
        }
        sample_losses[batch_index] = sample_loss;
    }

    size_t output_dims[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        output_dims[i] = 1;
    }

    if(no_reduction) {
        output_dims[batch_dim] = num_batches;
        Tensor output = create_tensor(output_dims);
        for(size_t i = 0; i < num_batches; i++) {
            output.data[i] = sample_losses[i];
        }
        free(sample_losses);
        return output;
    }
    else {
        Tensor output = create_tensor(output_dims);
        float total = 0.0f;
        for(size_t i = 0; i < num_batches; i++) {
            total += sample_losses[i];
        }
        if(mean_reduction) {
            output.data[0] = total / (float)num_batches;
        } else {
            output.data[0] = total;
        }
        free(sample_losses);
        return output;
    }
}

void categorical_cross_entropy_gradients(const Tensor prediction, const Tensor target, const Tensor upstream_gradients, Tensor* grad_prediction, Tensor* grad_target, float label_smoothing, size_t batch_dim, bool no_reduction, bool sum_reduction, bool mean_reduction) {
    size_t num_classes = prediction.dims[TENSOR_DIMENSIONS - 1];
    size_t num_batches = prediction.dims[batch_dim];
    float smoothing_per_class = label_smoothing / (float)num_classes;
    float target_scale = 1.0f - label_smoothing;

    for(size_t batch_index = 0; batch_index < num_batches; batch_index++) {
        size_t outer_offset = batch_index * num_classes;

        float upstream_gradient;
        if(no_reduction) {
            upstream_gradient = upstream_gradients.data[batch_index];
        } else if(mean_reduction) {
            upstream_gradient = upstream_gradients.data[0] / (float)num_batches;
        } else {
            upstream_gradient = upstream_gradients.data[0];
        }

        float max_prediction_value = prediction.data[outer_offset];
        for(size_t i = 1; i < num_classes; i++) {
            if(max_prediction_value < prediction.data[outer_offset + i]) {
                max_prediction_value = prediction.data[outer_offset + i];
            }
        }

        float exp_sum = 0.0f;
        for(size_t i = 0; i < num_classes; i++) {
            exp_sum += expf(prediction.data[outer_offset + i] - max_prediction_value);
        }
        float inverse_exp_sum = 1.0f / exp_sum;
        float log_exp_sum = logf(exp_sum);

        for(size_t i = 0; i < num_classes; i++) {
            float prediction_probability = (prediction.data[outer_offset + i] - max_prediction_value) - log_exp_sum;
            float target_probability = target.data[outer_offset + i];
            float probability = expf(prediction.data[outer_offset + i] - max_prediction_value) * inverse_exp_sum;
            float smoothed_target_probability = (target_scale * target_probability) + smoothing_per_class;

            grad_prediction->data[outer_offset + i] += (probability - smoothed_target_probability) * upstream_gradient;
            grad_target->data[outer_offset + i] += (-1.0f * (target_scale * prediction_probability)) * upstream_gradient;
        }
    }
}

Tensor mean_square_error(const Tensor prediction, const Tensor target, size_t batch_dim, bool no_reduction, bool sum_reduction, bool mean_reduction) {
    size_t num_batches = prediction.dims[batch_dim];
    size_t elements_in_batch = 1;
    for(size_t i = (batch_dim + 1); i < TENSOR_DIMENSIONS; i++) {
        elements_in_batch *= prediction.dims[i];
    }

    float* sample_losses = malloc(num_batches * sizeof(float));
    for(size_t batch_index = 0; batch_index < num_batches; batch_index++) {
        float sum = 0.0f;
        for(size_t i = 0; i < elements_in_batch; i++) {
            float x = prediction.data[(batch_index * elements_in_batch) + i] - target.data[(batch_index * elements_in_batch) + i];
            sum += x * x;
        }
        sample_losses[batch_index] = sum / (float)elements_in_batch;
    }

    size_t output_dims[TENSOR_DIMENSIONS];
    for(size_t i = 0; i < TENSOR_DIMENSIONS; i++) { output_dims[i] = 1; }

    if(no_reduction) {
        output_dims[batch_dim] = num_batches;
        Tensor output = create_tensor(output_dims);
        for(size_t i = 0; i < num_batches; i++) {
            output.data[i] = sample_losses[i];
        }
        free(sample_losses);
        return output;
    }
    else {
        Tensor output = create_tensor(output_dims);
        float total = 0.0f;
        for(size_t i = 0; i < num_batches; i++) {
            total += sample_losses[i];
        }
        output.data[0] = (mean_reduction) ? (total / (float)num_batches) : total;
        free(sample_losses);
        return output;
    }
}

void mean_square_error_gradients(const Tensor prediction, const Tensor target, const Tensor upstream_gradients, Tensor* grad_prediction, Tensor* grad_target, size_t batch_dim, bool no_reduction, bool sum_reduction, bool mean_reduction) {
    size_t num_batches = prediction.dims[batch_dim];
    size_t elements_in_batch = 1;
    for(size_t i = (batch_dim + 1); i < TENSOR_DIMENSIONS; i++) {
        elements_in_batch *= prediction.dims[i];
    }

    for(size_t batch_index = 0; batch_index < num_batches; batch_index++) {
        float upstream_gradient;
        if(no_reduction) {
            upstream_gradient = upstream_gradients.data[batch_index];
        } else if(mean_reduction) {
            upstream_gradient = upstream_gradients.data[0] / (float)num_batches;
        } else {
            upstream_gradient = upstream_gradients.data[0];
        }

        float scaled_gradient = (2.0f * upstream_gradient) / (float)elements_in_batch;
        for(size_t i = 0; i < elements_in_batch; i++) {
            float difference = prediction.data[(batch_index * elements_in_batch) + i] - target.data[(batch_index * elements_in_batch) + i];
            grad_prediction->data[(batch_index * elements_in_batch) + i] += scaled_gradient * difference;
            grad_target->data[(batch_index * elements_in_batch) + i] -= scaled_gradient * difference;
        }
    }
}

Tensor householder_qr(const Tensor input, bool flatten_to_2d) {
    size_t working_num_rows;
    size_t working_num_columns;
    size_t total_slices;
    size_t elements_per_slice;

    if (flatten_to_2d) {
        working_num_rows = input.dims[0];
        working_num_columns = input.total_elements / input.dims[0];
        total_slices = 1;
    }
    else {
        working_num_rows = input.dims[TENSOR_DIMENSIONS - 2];
        working_num_columns = input.dims[TENSOR_DIMENSIONS - 1];
        total_slices = 1;
        for (size_t d = 0; d < TENSOR_DIMENSIONS - 2; d++) {
            total_slices *= input.dims[d];
        }
    }
    elements_per_slice = working_num_rows * working_num_columns;

    size_t num_reflections = (working_num_rows < working_num_columns) ? working_num_rows : working_num_columns;

    size_t working_dims[TENSOR_DIMENSIONS];
    for (size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
        working_dims[i] = 1;
    }
    if (flatten_to_2d) {
        working_dims[TENSOR_DIMENSIONS - 2] = working_num_rows;
        working_dims[TENSOR_DIMENSIONS - 1] = working_num_columns;
    }
    else {
        for (size_t i = 0; i < TENSOR_DIMENSIONS; i++) {
            working_dims[i] = input.dims[i];
        }
    }

    size_t q_slice_size = working_num_rows * working_num_rows;
    float* Q = malloc(total_slices * q_slice_size * sizeof(float));
    float* R = malloc(total_slices * elements_per_slice * sizeof(float));
    memcpy(R, input.data, total_slices * elements_per_slice * sizeof(float));
    float* v = malloc(working_num_rows * sizeof(float));

    for (size_t s = 0; s < total_slices; s++) {
        size_t qoff = s * q_slice_size;
        size_t roff = s * elements_per_slice;
        for (size_t i = 0; i < q_slice_size; i++) {
            Q[qoff + i] = 0.0f;
        }
        for (size_t i = 0; i < working_num_rows; i++) {
            Q[qoff + i * working_num_rows + i] = 1.0f;
        }

        for (size_t k = 0; k < num_reflections; k++) {
            size_t sub_len = working_num_rows - k;

            for (size_t i = 0; i < sub_len; i++) {
                v[i] = R[roff + (k + i) * working_num_columns + k];
            }

            float norm_sq = 0.0f;
            for (size_t i = 0; i < sub_len; i++) {
                norm_sq += v[i] * v[i];
            }
            float norm = sqrtf(norm_sq);

            v[0] += (v[0] >= 0.0f) ? norm : -norm;

            float v_sq = 0.0f;
            for (size_t i = 0; i < sub_len; i++) {
                v_sq += v[i] * v[i];
            }
            if (v_sq < 1e-10f) {
                continue;
            }
            float scale = 2.0f / v_sq;

            for (size_t j = k; j < working_num_columns; j++) {
                float dot = 0.0f;
                for (size_t i = 0; i < sub_len; i++) {
                    dot += v[i] * R[roff + (k + i) * working_num_columns + j];
                }
                dot *= scale;
                for (size_t i = 0; i < sub_len; i++) {
                    R[roff + (k + i) * working_num_columns + j] -= dot * v[i];
                }
            }

            for (size_t row = 0; row < working_num_rows; row++) {
                float dot = 0.0f;
                for (size_t i = 0; i < sub_len; i++) {
                    dot += Q[qoff + row * working_num_rows + (k + i)] * v[i];
                }
                dot *= scale;
                for (size_t i = 0; i < sub_len; i++) {
                    Q[qoff + row * working_num_rows + (k + i)] -= dot * v[i];
                }
            }
        }
    }

    free(v);
    free(R);

    Tensor output = create_tensor(working_dims);
    set_tensor_data_to_zeros(&output);

    size_t copy_cols = (working_num_rows < working_num_columns) ? working_num_rows : working_num_columns;

    for (size_t s = 0; s < total_slices; s++) {
        size_t qoff = s * q_slice_size;
        size_t ooff = s * elements_per_slice;
        for (size_t row = 0; row < working_num_rows; row++) {
            for (size_t col = 0; col < copy_cols; col++) {
                output.data[ooff + row * working_num_columns + col] = Q[qoff + row * working_num_rows + col];
            }
        }
    }
    free(Q);

    if (flatten_to_2d) {
        Tensor reshaped = resize(output, input.dims);
        free_tensor(&output);
        return reshaped;
    }

    return output;
}

/* Autograd Engine */

typedef enum {
    OP_TYPE_INPUT,
    OP_TYPE_RECURRENT_INPUT,
    OP_TYPE_TARGET,
    OP_TYPE_HYPERPARAMETERS,
    OP_TYPE_PARAMETERS,
    OP_TYPE_RESIZE,
    OP_TYPE_CONCAT,
    OP_TYPE_ADD,
    OP_TYPE_SCALAR_TENSOR_MUL,
    OP_TYPE_ELEMENTWISE_MUL,
    OP_TYPE_MATMUL,
    OP_TYPE_CONVOLUTION,
    OP_TYPE_LAYER_NORM,
    OP_TYPE_RMS_NORM,
    OP_TYPE_SIGMOID,
    OP_TYPE_SOFTMAX,
    OP_TYPE_GELU,
    OP_TYPE_SWISH,
    OP_TYPE_MISH,
    OP_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS,
    OP_TYPE_MEAN_SQUARE_ERROR
} OperationType;

typedef enum {
    REDUCTION_TYPE_NONE,
    REDUCTION_TYPE_SUM,
    REDUCTION_TYPE_MEAN
} LossFunctionReductionType;

typedef enum {
    OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT,
    OPTIMIZER_ADAM,
    OPTIMIZER_MUON,
    OPTIMIZER_ANO
} OptimizationAlgorithm;

typedef struct {
    float learning_rate;
} StochasticGradientDescentConfig;

typedef struct {
    size_t* dims;
    bool allow_parameter_updates;
    OptimizationAlgorithm optimization_algorithm;
    void* optimizer_config;
} ParameterConfig;

typedef struct {
    size_t* dims;
} ResizeConfig;

typedef struct {
    size_t concat_dim;
} ConcatConfig;

typedef struct {
    size_t dim;
} SoftmaxConfig;

typedef struct {
    float epsilon;
    bool uses_batches;
    size_t batch_dim;
    bool allow_gamma_param_updates;
    bool allow_beta_param_updates;
    OptimizationAlgorithm gamma_optimization_algorithm;
    OptimizationAlgorithm beta_optimization_algorithm;
    void* gamma_optimizer_config;
    void* beta_optimizer_config;
} LayerNormConfig;

typedef struct {
    float epsilon;
    bool uses_batches;
    size_t batch_dim;
    bool allow_gamma_parameter_updates;
    OptimizationAlgorithm gamma_optimization_algorithm;
    void* gamma_optimizer_config;
} RMSNormConfig;

typedef struct {
    float label_smoothing;
    size_t batch_dim;
    LossFunctionReductionType reduction_type;
} CategoricalCrossEntropyConfig;

typedef struct {
    size_t batch_dim;
    LossFunctionReductionType reduction_type;
} MeanSquareErrorConfig;

typedef struct {
    size_t total_src_nodes;
    size_t total_dst_nodes;
    size_t* src_node_indices;
    size_t* dst_node_indices;
    OperationType op_type;
    void* config;
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

DirectedAcyclicGraph create_DirectedAcyclicGraph() {
    DirectedAcyclicGraph dag;
    dag.current_time_step = 1;
    dag.total_nodes = 0;
    dag.total_tape_entries = 0;
    dag.nodes = NULL;
    dag.tape_entries = NULL;
    return dag;
}

size_t add_node(DirectedAcyclicGraph* dag, OperationType op_type, void* config) {
    size_t node_index = dag->total_nodes;
    dag->total_nodes++;
    dag->nodes = realloc(dag->nodes, dag->total_nodes * sizeof(Node));

    dag->nodes[node_index].total_src_nodes = 0;
    dag->nodes[node_index].total_dst_nodes = 0;
    dag->nodes[node_index].src_node_indices = NULL;
    dag->nodes[node_index].dst_node_indices = NULL;
    dag->nodes[node_index].op_type = op_type;
    dag->nodes[node_index].config = config;
    dag->nodes[node_index].total_tape_entries = 0;
    dag->nodes[node_index].tape_entry_indices = NULL;

    if(op_type == OP_TYPE_PARAMETERS) {
        size_t tape_entry_index = dag->total_tape_entries;
        dag->total_tape_entries++;
        dag->tape_entries = realloc(dag->tape_entries, dag->total_tape_entries * sizeof(TapeEntry));
        dag->tape_entries[tape_entry_index].time_step = 1;
        dag->tape_entries[tape_entry_index].node_index = node_index;
        dag->tape_entries[tape_entry_index].total_src_tape_entries = 0;
        dag->tape_entries[tape_entry_index].total_dst_tape_entries = 0;
        dag->tape_entries[tape_entry_index].src_tape_entry_indices = NULL;
        dag->tape_entries[tape_entry_index].src_tape_entry_indices = NULL;
        ParameterConfig* parameter_config = dag->nodes[node_index].config;
        dag->tape_entries[tape_entry_index].output_tensor = create_tensor(parameter_config->dims);
        set_tensor_data_to_zeros(&dag->tape_entries[tape_entry_index].output_tensor);
        dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(NULL);

        dag->nodes[node_index].total_tape_entries = 1;
        dag->nodes[node_index].tape_entry_indices = malloc(sizeof(size_t));
        dag->nodes[node_index].tape_entry_indices[0] = tape_entry_index;
    }

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

void initialize_parameters(DirectedAcyclicGraph* dag, PCGRandomNumberGenerator* rng) {
    const float SQRT2 = 1.41421356237f;
    const int MAX_BFS_DEPTH = 3;

    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        if(dag->nodes[node_index].op_type != OP_TYPE_PARAMETERS) { continue; }
        if(dag->nodes[node_index].total_tape_entries == 0) { continue; }

        size_t tape_entry_index = dag->nodes[node_index].tape_entry_indices[0];

        if(dag->tape_entries[tape_entry_index].output_tensor.data == NULL || dag->tape_entries[tape_entry_index].output_tensor.total_elements == 0) { continue; }

        size_t num_rows = dag->tape_entries[tape_entry_index].output_tensor.dims[TENSOR_DIMENSIONS - 2];
        size_t num_columns = dag->tape_entries[tape_entry_index].output_tensor.dims[TENSOR_DIMENSIONS - 1];

        bool only_used_in_addition = true;
        bool only_used_in_matmul = true;
        bool only_used_in_conv = true;
        bool feeds_recurrent_input = false;
        bool is_gamma_parameters = false;
        bool is_beta_parameters = false;

        unsigned int closest_relu_like_function = UINT_MAX;
        unsigned int closest_sigmoid = UINT_MAX;
        unsigned int closest_softmax = UINT_MAX;
        unsigned int closest_cce_loss = UINT_MAX;
        unsigned int closest_mse_loss = UINT_MAX;

        for(size_t dst_node_index_inside_node = 0; dst_node_index_inside_node < dag->nodes[node_index].total_dst_nodes; dst_node_index_inside_node++) {
            size_t dst_node_index = dag->nodes[node_index].dst_node_indices[dst_node_index_inside_node];
            if(dag->nodes[dst_node_index].op_type != OP_TYPE_ADD) { only_used_in_addition = false; }
            if(dag->nodes[dst_node_index].op_type != OP_TYPE_MATMUL) { only_used_in_matmul = false; }
            if(dag->nodes[dst_node_index].op_type != OP_TYPE_CONVOLUTION) { only_used_in_conv = false; }
            if(dag->nodes[dst_node_index].op_type == OP_TYPE_LAYER_NORM) {
                if(dag->nodes[dst_node_index].src_node_indices[1] == node_index) { is_gamma_parameters = true; }
                if(dag->nodes[dst_node_index].src_node_indices[2] == node_index) { is_beta_parameters = true; }
                continue;
            }
            if(dag->nodes[dst_node_index].op_type == OP_TYPE_RMS_NORM) {
                if(dag->nodes[dst_node_index].src_node_indices[1] == node_index) { is_gamma_parameters = true; }
                continue;
            }

            size_t queue_head = 0;
            size_t queue_tail = 0;
            size_t* next_node_indices_queue = malloc(dag->total_nodes * sizeof(size_t));
            unsigned int* depths_queue = malloc(dag->total_nodes * sizeof(unsigned int));
            uint8_t* already_visited_arr = calloc(dag->total_nodes, sizeof(uint8_t));

            next_node_indices_queue[queue_tail] = dst_node_index;
            depths_queue[queue_tail] = 1;
            already_visited_arr[dst_node_index] = 1;
            queue_tail++;

            while(queue_head < queue_tail) {
                size_t current_node_index = next_node_indices_queue[queue_head];
                unsigned int current_depth = depths_queue[queue_head];
                queue_head++;

                if(current_depth > (unsigned int)MAX_BFS_DEPTH) { continue; }

                OperationType op = dag->nodes[current_node_index].op_type;
                if((op == OP_TYPE_GELU || op == OP_TYPE_SWISH || op == OP_TYPE_MISH) && (current_depth < closest_relu_like_function)) {
                    closest_relu_like_function = current_depth;
                }
                else if(op == OP_TYPE_SIGMOID && (current_depth < closest_sigmoid)) {
                    closest_sigmoid = current_depth;
                }
                else if(op == OP_TYPE_SOFTMAX && (current_depth < closest_softmax)) {
                    closest_softmax = current_depth;
                }
                else if(op == OP_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS && (current_depth < closest_cce_loss)) {
                    closest_cce_loss = current_depth;
                }
                else if(op == OP_TYPE_MEAN_SQUARE_ERROR && (current_depth < closest_mse_loss)) {
                    closest_mse_loss = current_depth;
                }
                else if(op == OP_TYPE_RECURRENT_INPUT) {
                    feeds_recurrent_input = true;
                }

                if(current_depth == (unsigned int)MAX_BFS_DEPTH) { continue; }

                for(size_t j = 0; j < dag->nodes[current_node_index].total_dst_nodes; j++) {
                    size_t next_node_index = dag->nodes[current_node_index].dst_node_indices[j];

                    if(already_visited_arr[next_node_index]) {
                        continue;
                    }

                    next_node_indices_queue[queue_tail] = next_node_index;
                    depths_queue[queue_tail] = current_depth + 1;
                    already_visited_arr[next_node_index] = 1;
                    queue_tail++;
                }
            }
            free(next_node_indices_queue);
            free(depths_queue);
            free(already_visited_arr);

            queue_head = 0;
            queue_tail = 0;
            size_t* prev_node_indices_queue = malloc(dag->total_nodes * sizeof(size_t));
            unsigned int* back_depths_queue = malloc(dag->total_nodes * sizeof(unsigned int));
            uint8_t* back_visited_arr = calloc(dag->total_nodes, sizeof(uint8_t));

            prev_node_indices_queue[queue_tail] = dst_node_index;
            back_depths_queue[queue_tail] = 1;
            back_visited_arr[dst_node_index] = 1;
            queue_tail++;

            while(queue_head < queue_tail) {
                size_t current_node_index = prev_node_indices_queue[queue_head];
                unsigned int current_depth = back_depths_queue[queue_head];
                queue_head++;

                if(current_depth > (unsigned int)MAX_BFS_DEPTH) { continue; }

                if(dag->nodes[current_node_index].op_type == OP_TYPE_RECURRENT_INPUT) {
                    feeds_recurrent_input = true;
                }

                if(current_depth == (unsigned int)MAX_BFS_DEPTH) { continue; }

                for(size_t j = 0; j < dag->nodes[current_node_index].total_src_nodes; j++) {
                    size_t prev_node_index = dag->nodes[current_node_index].src_node_indices[j];

                    if(back_visited_arr[prev_node_index]) { continue; }

                    prev_node_indices_queue[queue_tail] = prev_node_index;
                    back_depths_queue[queue_tail] = current_depth + 1;
                    back_visited_arr[prev_node_index] = 1;
                    queue_tail++;
                }
            }
            free(prev_node_indices_queue);
            free(back_depths_queue);
            free(back_visited_arr);
        }

        if(only_used_in_addition || is_beta_parameters) {
            continue;
        }
        else if(is_gamma_parameters) {
            set_tensor_data_to_ones(&dag->tape_entries[tape_entry_index].output_tensor);
        }
        else {
            float fan_in = num_rows;
            float fan_out = num_columns;

            if(only_used_in_conv) {
                size_t out_channels = dag->tape_entries[tape_entry_index].output_tensor.dims[0];
                size_t in_channels = dag->tape_entries[tape_entry_index].output_tensor.dims[1];
                size_t kernel_extent = dag->tape_entries[tape_entry_index].output_tensor.dims[TENSOR_DIMENSIONS - 1];
                fan_in = (float)in_channels * kernel_extent;
                fan_out = (float)out_channels * kernel_extent;
            }

            float std_dev;
            if(closest_relu_like_function != UINT_MAX) {
                std_dev = SQRT2 / sqrtf(fan_in);
            }
            else {
                std_dev = sqrtf(2.0f / (fan_in + fan_out));
            }

            if((closest_softmax != UINT_MAX) || (closest_cce_loss != UINT_MAX)) {
                std_dev = sqrtf(1.0f / fan_in);
            }

            add_gaussian_noise_to_tensor_data(&dag->tape_entries[tape_entry_index].output_tensor, 0.0f, std_dev, rng);

            if((only_used_in_matmul || feeds_recurrent_input) && (num_rows >= num_columns)) {
                Tensor orthonormal_tensor = householder_qr(dag->tape_entries[tape_entry_index].output_tensor, false);
                for(size_t i = 0; i < dag->tape_entries[tape_entry_index].output_tensor.total_elements; i++) {
                    dag->tape_entries[tape_entry_index].output_tensor.data[i] = orthonormal_tensor.data[i];
                }
                free_tensor(&orthonormal_tensor);
            }
            else if(only_used_in_conv) {
                size_t flat_rows = dag->tape_entries[tape_entry_index].output_tensor.dims[0];
                size_t flat_cols = dag->tape_entries[tape_entry_index].output_tensor.total_elements / flat_rows;
                if(flat_rows >= flat_cols) {
                    Tensor orthonormal_tensor = householder_qr(dag->tape_entries[tape_entry_index].output_tensor, true);
                    for(size_t i = 0; i < dag->tape_entries[tape_entry_index].output_tensor.total_elements; i++) {
                        dag->tape_entries[tape_entry_index].output_tensor.data[i] = orthonormal_tensor.data[i];
                    }
                    free_tensor(&orthonormal_tensor);
                }
            }
        }
    }
}

size_t add_input(DirectedAcyclicGraph* dag, size_t input_node_index, const Tensor input) {
    size_t tape_entry_index = dag->total_tape_entries;
    dag->total_tape_entries++;
    dag->tape_entries = realloc(dag->tape_entries, dag->total_tape_entries * sizeof(TapeEntry));

    dag->tape_entries[tape_entry_index].time_step = dag->current_time_step;
    dag->tape_entries[tape_entry_index].node_index = input_node_index;
    dag->tape_entries[tape_entry_index].total_src_tape_entries = 0;
    dag->tape_entries[tape_entry_index].total_dst_tape_entries = 0;
    dag->tape_entries[tape_entry_index].src_tape_entry_indices = NULL;
    dag->tape_entries[tape_entry_index].dst_tape_entry_indices = NULL;
    dag->tape_entries[tape_entry_index].output_tensor.total_elements = input.total_elements;
    dag->tape_entries[tape_entry_index].output_tensor.dims = input.dims;
    dag->tape_entries[tape_entry_index].output_tensor.data = input.data;
    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(NULL);

    size_t tape_entry_index_inside_node = dag->nodes[input_node_index].total_tape_entries;
    dag->nodes[input_node_index].total_tape_entries++;
    dag->nodes[input_node_index].tape_entry_indices = realloc(dag->nodes[input_node_index].tape_entry_indices, dag->nodes[input_node_index].total_tape_entries * sizeof(size_t));
    dag->nodes[input_node_index].tape_entry_indices[tape_entry_index_inside_node] = tape_entry_index;
    return tape_entry_index;
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
            if(dag.nodes[src_node_index].op_type == OP_TYPE_PARAMETERS) { continue; }
            size_t src_tape_entry_index = dag.nodes[src_node_index].tape_entry_indices[dag.nodes[src_node_index].total_tape_entries - 1];
            if(dag.tape_entries[src_tape_entry_index].time_step != dag.current_time_step) { return false; }
        }
        else { return false; }
    }
    return true;
}

size_t add_tape_entry(DirectedAcyclicGraph* dag, size_t node_index) {
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

    if(dag->nodes[node_index].op_type == OP_TYPE_RECURRENT_INPUT) {
        size_t src_node_index = dag->nodes[node_index].src_node_indices[0];
        size_t src_tape_entry_index_inside_node = dag->nodes[src_node_index].total_tape_entries - 1;
        size_t src_tape_entry_index = dag->nodes[src_node_index].tape_entry_indices[src_tape_entry_index_inside_node];
        dag->tape_entries[tape_entry_index].output_tensor = deep_copy_tensor(dag->tape_entries[src_tape_entry_index].output_tensor);
        dag->tape_entries[tape_entry_index].time_step++;
    }

    return tape_entry_index;
}

static inline void forward(DirectedAcyclicGraph* dag) {
    size_t tape_entry_index = 0;
    while(tape_entry_index < dag->total_tape_entries) {
        if(dag->tape_entries[tape_entry_index].time_step == dag->current_time_step) {
            size_t node_index = dag->tape_entries[tape_entry_index].node_index;
            switch(dag->nodes[node_index].op_type) {
                case OP_TYPE_RESIZE: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    ResizeConfig* resize_config = dag->nodes[node_index].config;
                    dag->tape_entries[tape_entry_index].output_tensor = resize(dag->tape_entries[src_tape_entry_index].output_tensor, resize_config->dims);
                    break;
                }
                case OP_TYPE_CONCAT: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    ConcatConfig* concat_config = dag->nodes[node_index].config;
                    dag->tape_entries[tape_entry_index].output_tensor = concat(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, concat_config->concat_dim);
                    break;
                }
                case OP_TYPE_ADD: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = add(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OP_TYPE_SCALAR_TENSOR_MUL: {
                    size_t scalar_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t tensor_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = scalar_tensor_mul(dag->tape_entries[scalar_tape_entry_index].output_tensor, dag->tape_entries[tensor_tape_entry_index].output_tensor);
                    break;
                }
                case OP_TYPE_ELEMENTWISE_MUL: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = elementwise_mul(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OP_TYPE_MATMUL: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = matmul(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OP_TYPE_CONVOLUTION: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    dag->tape_entries[tape_entry_index].output_tensor = convolution(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor);
                    break;
                }
                case OP_TYPE_LAYER_NORM: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    size_t src_tape_entry_index2 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[2];
                    LayerNormConfig* layer_norm_config = dag->nodes[node_index].config;
                    dag->tape_entries[tape_entry_index].output_tensor = layer_norm(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[src_tape_entry_index2].output_tensor, layer_norm_config->epsilon, layer_norm_config->uses_batches, layer_norm_config->batch_dim);
                    break;
                }
                case OP_TYPE_RMS_NORM: {
                    size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    RMSNormConfig* rms_norm_config = dag->nodes[node_index].config;
                    dag->tape_entries[tape_entry_index].output_tensor = rms_norm(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, rms_norm_config->epsilon, rms_norm_config->uses_batches, rms_norm_config->batch_dim);
                    break;
                }
                case OP_TYPE_SIGMOID: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = sigmoid(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OP_TYPE_SOFTMAX: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    SoftmaxConfig* softmax_config = dag->nodes[node_index].config;
                    dag->tape_entries[tape_entry_index].output_tensor = softmax(dag->tape_entries[src_tape_entry_index].output_tensor, softmax_config->dim);
                    break;
                }
                case OP_TYPE_GELU: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = gelu(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OP_TYPE_SWISH: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = swish(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OP_TYPE_MISH: {
                    size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    dag->tape_entries[tape_entry_index].output_tensor = mish(dag->tape_entries[src_tape_entry_index].output_tensor);
                    break;
                }
                case OP_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS: {
                    size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    CategoricalCrossEntropyConfig* cce_config = dag->nodes[node_index].config;
                    bool no_reduction = (cce_config->reduction_type == REDUCTION_TYPE_NONE) ? 1 : 0;
                    bool sum_reduction = (cce_config->reduction_type == REDUCTION_TYPE_SUM) ? 1 : 0;;
                    bool mean_reduction = (cce_config->reduction_type == REDUCTION_TYPE_MEAN) ? 1 : 0;
                    dag->tape_entries[tape_entry_index].output_tensor = categorical_cross_entropy(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, cce_config->label_smoothing, cce_config->batch_dim, no_reduction, sum_reduction, mean_reduction);
                    break;
                }
                case OP_TYPE_MEAN_SQUARE_ERROR: {
                    size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                    size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                    MeanSquareErrorConfig* mse_config = dag->nodes[node_index].config;
                    bool no_reduction = (mse_config->reduction_type == REDUCTION_TYPE_NONE) ? 1 : 0;
                    bool sum_reduction = (mse_config->reduction_type == REDUCTION_TYPE_SUM) ? 1 : 0;;
                    bool mean_reduction = (mse_config->reduction_type == REDUCTION_TYPE_MEAN) ? 1 : 0;
                    dag->tape_entries[tape_entry_index].output_tensor = mean_square_error(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, mse_config->batch_dim, no_reduction, sum_reduction, mean_reduction);
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
    dag->current_time_step++;
}

void backward(DirectedAcyclicGraph* dag) {
    for(size_t tape_entry_index = dag->total_tape_entries; tape_entry_index-- > 0;) {
        for(size_t i = 0; i < dag->tape_entries[tape_entry_index].total_src_tape_entries; i++) {
            size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[i];
            if(dag->tape_entries[src_tape_entry_index].gradient_tensor.data == NULL) {
                dag->tape_entries[src_tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[src_tape_entry_index].output_tensor.dims);
                set_tensor_data_to_zeros(&dag->tape_entries[src_tape_entry_index].gradient_tensor);
            }
        }
        size_t node_index = dag->tape_entries[tape_entry_index].node_index;
        switch(dag->nodes[node_index].op_type) {
            case OP_TYPE_RESIZE: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                resize_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OP_TYPE_CONCAT: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                ConcatConfig* concat_config = dag->nodes[node_index].config;
                concat_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor, concat_config->concat_dim);
                break;
            }
            case OP_TYPE_ADD: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                add_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OP_TYPE_SCALAR_TENSOR_MUL: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                scalar_tensor_mul_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OP_TYPE_ELEMENTWISE_MUL: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                elementwise_mul_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OP_TYPE_MATMUL: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                matmul_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OP_TYPE_CONVOLUTION: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                convolution_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor);
                break;
            }
            case OP_TYPE_LAYER_NORM: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                size_t src_tape_entry_index2 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[2];
                LayerNormConfig* layer_norm_config = dag->nodes[node_index].config;
                layer_norm_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, dag->tape_entries[src_tape_entry_index2].output_tensor, layer_norm_config->epsilon, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor, &dag->tape_entries[src_tape_entry_index2].gradient_tensor, layer_norm_config->uses_batches, layer_norm_config->batch_dim);
                break;
            }
            case OP_TYPE_RMS_NORM: {
                size_t src_tape_entry_index0 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t src_tape_entry_index1 = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                RMSNormConfig* rms_norm_config = dag->nodes[node_index].config;
                rms_norm_gradients(dag->tape_entries[src_tape_entry_index0].output_tensor, dag->tape_entries[src_tape_entry_index1].output_tensor, rms_norm_config->epsilon, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index0].gradient_tensor, &dag->tape_entries[src_tape_entry_index1].gradient_tensor, rms_norm_config->uses_batches, rms_norm_config->batch_dim);
                break;
            }
            case OP_TYPE_SIGMOID: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                sigmoid_gradients(dag->tape_entries[tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OP_TYPE_SOFTMAX: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                SoftmaxConfig* softmax_config = dag->nodes[node_index].config;
                softmax_gradients(dag->tape_entries[tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor, softmax_config->dim);
                break;
            }
            case OP_TYPE_GELU: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                gelu_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OP_TYPE_SWISH: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                swish_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OP_TYPE_MISH: {
                size_t src_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                mish_gradients(dag->tape_entries[src_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[src_tape_entry_index].gradient_tensor);
                break;
            }
            case OP_TYPE_CATEGORICAL_CROSS_ENTROPY_LOSS: {
                if(dag->tape_entries[tape_entry_index].total_dst_tape_entries == 0 && dag->tape_entries[tape_entry_index].gradient_tensor.data == NULL) {
                    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[tape_entry_index].output_tensor.dims);
                    set_tensor_data_to_ones(&dag->tape_entries[tape_entry_index].gradient_tensor);
                }
                size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                CategoricalCrossEntropyConfig* cce_config = dag->nodes[node_index].config;
                bool no_reduction = (cce_config->reduction_type == REDUCTION_TYPE_NONE) ? 1 : 0;
                bool sum_reduction = (cce_config->reduction_type == REDUCTION_TYPE_SUM) ? 1 : 0;;
                bool mean_reduction = (cce_config->reduction_type == REDUCTION_TYPE_MEAN) ? 1 : 0;
                categorical_cross_entropy_gradients(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[prediction_tape_entry_index].gradient_tensor, &dag->tape_entries[target_tape_entry_index].gradient_tensor, cce_config->label_smoothing, cce_config->batch_dim, no_reduction, sum_reduction, mean_reduction);
                break;
            }
            case OP_TYPE_MEAN_SQUARE_ERROR: {
                if(dag->tape_entries[tape_entry_index].total_dst_tape_entries == 0 && dag->tape_entries[tape_entry_index].gradient_tensor.data == NULL) {
                    dag->tape_entries[tape_entry_index].gradient_tensor = create_tensor(dag->tape_entries[tape_entry_index].output_tensor.dims);
                    set_tensor_data_to_ones(&dag->tape_entries[tape_entry_index].gradient_tensor);
                }
                size_t prediction_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[0];
                size_t target_tape_entry_index = dag->tape_entries[tape_entry_index].src_tape_entry_indices[1];
                MeanSquareErrorConfig* mse_config = dag->nodes[node_index].config;
                bool no_reduction = (mse_config->reduction_type == REDUCTION_TYPE_NONE) ? 1 : 0;
                bool sum_reduction = (mse_config->reduction_type == REDUCTION_TYPE_SUM) ? 1 : 0;;
                bool mean_reduction = (mse_config->reduction_type == REDUCTION_TYPE_MEAN) ? 1 : 0;
                mean_square_error_gradients(dag->tape_entries[prediction_tape_entry_index].output_tensor, dag->tape_entries[target_tape_entry_index].output_tensor, dag->tape_entries[tape_entry_index].gradient_tensor, &dag->tape_entries[prediction_tape_entry_index].gradient_tensor, &dag->tape_entries[target_tape_entry_index].gradient_tensor, mse_config->batch_dim, no_reduction, sum_reduction, mean_reduction);
                break;
            }
            default: { break; }
        }
    }
}

void update_parameters(DirectedAcyclicGraph* dag) {
    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        if(dag->nodes[node_index].op_type == OP_TYPE_PARAMETERS) {
            ParameterConfig* param_config = dag->nodes[node_index].config;
            if(param_config->allow_parameter_updates) {
                if(param_config->optimization_algorithm == OPTIMIZER_STOCHASTIC_GRADIENT_DESCENT) {
                    StochasticGradientDescentConfig* sgd_config = param_config->optimizer_config;
                    size_t tape_entry_index = dag->nodes[node_index].tape_entry_indices[0];
                    for(size_t i = 0; i < dag->tape_entries[tape_entry_index].output_tensor.total_elements; i++) {
                        dag->tape_entries[tape_entry_index].output_tensor.data[i] -= dag->tape_entries[tape_entry_index].gradient_tensor.data[i] * sgd_config->learning_rate;
                    }
                }
            }
        }
    }
}

void reset_memory(DirectedAcyclicGraph* dag, bool keep_last_recurrent_input) {
    dag->current_time_step = 1;

    size_t preserved_tape_entries_capacity = 0;
    for(size_t i = 0; i < dag->total_nodes; i++) {
        if((dag->nodes[i].op_type == OP_TYPE_PARAMETERS || dag->nodes[i].op_type == OP_TYPE_HYPERPARAMETERS || (dag->nodes[i].op_type == OP_TYPE_RECURRENT_INPUT && keep_last_recurrent_input)) && dag->nodes[i].total_tape_entries > 0) {
            preserved_tape_entries_capacity++;
        }
    }
    TapeEntry* preserved_tape_entries = malloc(preserved_tape_entries_capacity * sizeof(TapeEntry));
    size_t preserved_tape_entries_index = 0;

    for(size_t tape_entry_index = 0; tape_entry_index < dag->total_tape_entries; tape_entry_index++) {
        free(dag->tape_entries[tape_entry_index].src_tape_entry_indices);
        free(dag->tape_entries[tape_entry_index].dst_tape_entry_indices);
        dag->tape_entries[tape_entry_index].src_tape_entry_indices = NULL;
        dag->tape_entries[tape_entry_index].dst_tape_entry_indices = NULL;

        size_t node_index = dag->tape_entries[tape_entry_index].node_index;
        bool preserve_tape_entry = false;
        if(dag->nodes[node_index].op_type == OP_TYPE_HYPERPARAMETERS || dag->nodes[node_index].op_type == OP_TYPE_PARAMETERS || (dag->nodes[node_index].op_type == OP_TYPE_RECURRENT_INPUT && keep_last_recurrent_input)) {
            size_t last_tape_entry_index_inside_node = dag->nodes[node_index].total_tape_entries - 1;
            size_t last_tape_entry_index = dag->nodes[node_index].tape_entry_indices[last_tape_entry_index_inside_node];
            if(last_tape_entry_index == tape_entry_index) {
                preserve_tape_entry = true;
            }
        }
        bool is_externally_owned_tensor = false;
        if(dag->nodes[node_index].op_type == OP_TYPE_INPUT || dag->nodes[node_index].op_type == OP_TYPE_TARGET) { is_externally_owned_tensor = true; }

        if(preserve_tape_entry) {
            preserved_tape_entries[preserved_tape_entries_index].time_step = 1;
            preserved_tape_entries[preserved_tape_entries_index].node_index = dag->tape_entries[tape_entry_index].node_index;
            preserved_tape_entries[preserved_tape_entries_index].total_src_tape_entries = 0;
            preserved_tape_entries[preserved_tape_entries_index].total_dst_tape_entries = 0;
            preserved_tape_entries[preserved_tape_entries_index].src_tape_entry_indices = NULL;
            preserved_tape_entries[preserved_tape_entries_index].dst_tape_entry_indices = NULL;
            preserved_tape_entries[preserved_tape_entries_index].output_tensor.total_elements = dag->tape_entries[tape_entry_index].output_tensor.total_elements;
            preserved_tape_entries[preserved_tape_entries_index].output_tensor.dims = dag->tape_entries[tape_entry_index].output_tensor.dims;
            preserved_tape_entries[preserved_tape_entries_index].output_tensor.data = dag->tape_entries[tape_entry_index].output_tensor.data;
            preserved_tape_entries[preserved_tape_entries_index].gradient_tensor = create_tensor(NULL);
            preserved_tape_entries_index++;
            free_tensor(&dag->tape_entries[tape_entry_index].gradient_tensor);
        }
        else {
            if(!is_externally_owned_tensor) {
                free_tensor(&dag->tape_entries[tape_entry_index].output_tensor);
            }
            free_tensor(&dag->tape_entries[tape_entry_index].gradient_tensor);
        }
    }

    free(dag->tape_entries);
    dag->tape_entries = preserved_tape_entries;
    dag->total_tape_entries = preserved_tape_entries_capacity;

    for(size_t node_index = 0; node_index < dag->total_nodes; node_index++) {
        free(dag->nodes[node_index].tape_entry_indices);
        dag->nodes[node_index].tape_entry_indices = NULL;
        dag->nodes[node_index].total_tape_entries = 0;

        for(size_t tape_entry_index = 0; tape_entry_index < dag->total_tape_entries; tape_entry_index++) {
            if(node_index == dag->tape_entries[tape_entry_index].node_index) {
                dag->nodes[node_index].total_tape_entries = 1;
                dag->nodes[node_index].tape_entry_indices = malloc(sizeof(size_t));
                dag->nodes[node_index].tape_entry_indices[0] = tape_entry_index;
                break;
            }
        }
    }
}
