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

Matrix * matrix_add(Matrix* m1, Matrix* m2);

Matrix * matrix_trans(Matrix* m);

Matrix * matrix_add_vector(Matrix* m, Matrix* v);

void print_matrix(Matrix* m);

double sigmoid(double v);

#endif
