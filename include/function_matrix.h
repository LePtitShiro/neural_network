#ifndef FUNCTION_MATRIX
#define FUNCTION_MATRIX


typedef struct{
    int rows;
    int cols;
    double* data;
} Matrix;

Matrix* create_matrix(int rows, int cols);

void free_matrix(Matrix* m);

Matrix * matrix_product(Matrix* m1, Matrix* m2);

Matrix * matrix_product_hadamard(Matrix* m1, Matrix* m2);

Matrix * matrix_add(Matrix* m1, Matrix* m2);

Matrix * matrix_sub(Matrix* m1, Matrix* m2);

Matrix * matrix_trans(Matrix* m);

Matrix * matrix_add_vector(Matrix* m, Matrix* v);

Matrix * matrix_copy(Matrix * m);

void print_matrix(Matrix* m);

double sigmoid(double v);

double derivate_sigmoid(double v);

double mse_cost(Matrix* m1, Matrix* m2);

Matrix* mse_derivate(Matrix* m1, Matrix* m2);

Matrix * matrix_scalar_multiply(Matrix* m1, double v);

#endif
