
#include "function_matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

Matrix* create_matrix(int rows, int cols)
{
    Matrix * m = (Matrix *)malloc(sizeof(Matrix));
    m->cols = cols;
    m->rows = rows;
    m->data = (double*)calloc(rows*cols,sizeof(double)); // On le rempli de 0
    return m;
}

void free_matrix(Matrix *m)
{
    if(m == NULL) return;
    free(m->data);
    free(m);
}


Matrix * matrix_product(Matrix * m1, Matrix * m2)
{
    if (m1 == NULL || m2 == NULL) return NULL;

    if (m1->cols != m2->rows){
        perror("Dimension error during product (standard)");
        return NULL;
    }

    Matrix * matrix_p = create_matrix(m1->rows,m2->cols);
    for (int i = 0; i < m1->rows;i++){
        for(int k = 0; k < m2->rows; k++){
            for(int j = 0; j < m2->cols; j++){
                matrix_p->data[i * m2->cols + j] += m1->data[i * m1->cols + k] * m2->data[k * m2->cols + j];
            }
        }
    }
    return matrix_p;
}

Matrix * matrix_product_hadamard(Matrix* m1, Matrix* m2)
{
    if (m1 == NULL || m2 == NULL) return NULL;

    if (m1->rows != m2->rows || m1->cols != m2->cols){
        perror("Dimension error during product (hadamard)");
        return NULL;
    }

    Matrix * new_matrix = create_matrix(m1->rows, m1->cols);

    for(int i = 0; i < (m1->rows * m1->cols); i++){
            new_matrix->data[i] = m1->data[i] * m2->data[i];

    }

    return new_matrix;
}


Matrix * matrix_add(Matrix* m1, Matrix* m2)
{
    if (m1 == NULL || m2 == NULL) return NULL;

    if (m1->rows != m2->rows || m1->cols != m2->cols){
        perror("Dimension error during addition");
        return NULL;
    }

    Matrix * new_matrix = create_matrix(m1->rows, m1->cols);

    for(int i = 0; i < (m1->rows * m1->cols); i++){
            new_matrix->data[i] = m1->data[i] + m2->data[i];

    }

    return new_matrix;
}

Matrix * matrix_sub(Matrix* m1, Matrix* m2)
{
    if (m1->rows != m2->rows || m1->cols != m2->cols){
        perror("Dimension error during substraction");
        return NULL;
    }

    Matrix * new_matrix = create_matrix(m1->rows, m1->cols);

    for(int i = 0; i < (m1->rows * m1->cols); i++){
            new_matrix->data[i] = m1->data[i] - m2->data[i];

    }

    return new_matrix;
}

Matrix * matrix_trans(Matrix *m)
{
    if(m == NULL) return NULL;

    Matrix * new_matrix = create_matrix(m->cols, m->rows);
    if(new_matrix == NULL) return NULL;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            int index = i*m->cols+j;
            int index2 = j*new_matrix->cols + i;
            new_matrix->data[index2] = m->data[index];
        }
    }

    return new_matrix;
}

Matrix * matrix_add_vector(Matrix* m, Matrix* v)
{
    if(m==NULL || v == NULL) return NULL;
    if(v->rows != 1 || m->cols != v->cols){
        perror("Dimension error during addition (vector)");
        return NULL;
    }

    Matrix * new_matrix = create_matrix(m->rows, m->cols);
    if(new_matrix == NULL) return NULL;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            int index = i*m->cols +j;
            new_matrix->data[index] = m->data[index] + v->data[j];
        }
    }
    return new_matrix;

}


void print_matrix(Matrix* m)
{
    if (m == NULL) {perror("m is null (print)"); return;}


    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            printf("%8.2f",m->data[i*m->cols+j]);
        }
        printf("\n");
    }
    printf("\n");
}


double sigmoid(double v)
{
    return 1 / (1 + exp(-v));
}

double derivate_sigmoid(double v)
{
    return v * (1 - v);
}

double mse_cost(Matrix* m1, Matrix* m2)
{
    if(m1 == NULL || m2 == NULL) return -1;

    if(m1->rows != m2->rows || m1->cols != m2->cols){
        perror("Dimension error during cost calculation");
        return -1;
    }

    double cost = 0;

    for(int i = 0; i < (m1->rows * m1->cols); i++){
        double temp = m1->data[i] - m2->data[i];
        cost += (temp*temp);
    }

    return cost / (m1->rows * m1->cols);
}

Matrix* mse_derivate(Matrix* m1, Matrix* m2)
{
    if(m1 == NULL || m2 == NULL) return NULL;
    if(m1->rows != m2->rows || m1->cols != m2->cols){
        perror("Dimension error during derivate calculation");
        return NULL;
    }

    return matrix_sub(m1,m2); //TODO : On peut ajouter *2 pour respecter la dérivé mais inutile en vrai
}

Matrix * matrix_copy(Matrix * m)
{
    if (m == NULL) return NULL;

    Matrix * new_matrix = create_matrix(m->rows, m->cols);
    if (m == NULL) return NULL;

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            new_matrix->data[i*m->cols+j] = m->data[i*m->cols +j];
        }
    }
    return new_matrix;
}


Matrix * matrix_scalar_multiply(Matrix* m1, double v)
{
    if(m1 == NULL){ perror("matrix is null (scalar_multiply)"); return NULL;}
    Matrix * new_matrix = create_matrix(m1->rows, m1->cols);
    for(int i = 0; i < m1->rows; i++){
        for(int j = 0; j < m1->cols; j++){
            new_matrix->data[i*m1->cols + j] = m1->data[i*m1->cols+j]*v;
        }
    }
    return new_matrix;

}
