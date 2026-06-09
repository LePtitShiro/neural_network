#include "network.h"
#include "function_matrix.h"
#include <math.h>
#include <stdlib.h>

void init_weights(Matrix * m, int e, int s)
{
    double x_value = sqrt(6/(double)(e+s));

    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            double v = random_double(-x_value,x_value);
            m->data[i*m->cols+j] = v;
        }
    }


}

double random_double(double a, double b)
{
    return a + ((double)rand() / (double)RAND_MAX)*(b-a);
}


void init_bias(Matrix * m)
{
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            m->data[i*m->cols+j] = 0;
        }
    }
}


Matrix * apply_sigmoid(Matrix *m)
{
    if(m == NULL) return NULL;

    Matrix * new_matrix = create_matrix(m->rows,m->cols);
    if(new_matrix == NULL) return NULL;


    for(int i = 0; i < new_matrix->rows; i++){
        for(int j = 0; j <new_matrix->cols; j++){
            int index = i*new_matrix->cols+j;
            new_matrix->data[index] = sigmoid(m->data[index]);
        }
    }
    return new_matrix;
}

void network_predict(Network *n, Matrix *input)
{
    Matrix * current_input = input;


    for(int i = 0; i < n->number_layers; i++){
        Matrix * m_product = matrix_product(n->layers[i].weights,current_input);
        Matrix * m_add = matrix_add(m_product,n->layers[i].bias);
        free(m_product);

        Matrix * m_sigmoid = apply_sigmoid(m_add);
        if(n->layers[i].output != NULL) free_matrix(n->layers[i].output);

        n->layers[i].output = m_sigmoid;

        current_input = m_sigmoid;

    }

}
