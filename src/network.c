#include "network.h"
#include "function_matrix.h"
#include <linux/limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        free_matrix(m_product);

        Matrix * m_sigmoid = apply_sigmoid(m_add);
        free_matrix(m_add);

        if(n->layers[i].output != NULL) free_matrix(n->layers[i].output);

        n->layers[i].output = m_sigmoid;

        current_input = m_sigmoid;


    }

}


void network_backward(Network *n, Matrix * goal, Matrix * input)
{
    int index_last_layer = n->number_layers-1;
    Matrix * output_last_layer = n->layers[index_last_layer].output;
    Matrix * m_BE = mse_derivate(output_last_layer,goal);
    Matrix * m_D = apply_derivate_sigmoid(output_last_layer);
    Matrix * delta = matrix_product_hadamard(m_D, m_BE);

    free_matrix(m_BE);
    free_matrix(m_D);

    if(n->layers[index_last_layer].delta != NULL ) free_matrix(n->layers[index_last_layer].delta);
    n->layers[index_last_layer].delta = delta;

    int i = index_last_layer - 1;
    for (int j = i; j >= 0 ; j--){
        Matrix * weight_next_trans = matrix_trans(n->layers[j+1].weights);
        Matrix * p_w = matrix_product(weight_next_trans, n->layers[j+1].delta);
        Matrix * deriv_sigmoid = apply_derivate_sigmoid(n->layers[j].output);


        if(n->layers[j].delta != NULL ) free_matrix(n->layers[j].delta);
        n->layers[j].delta = matrix_product_hadamard(deriv_sigmoid, p_w);


        free_matrix(weight_next_trans);
        free_matrix(p_w);
        free_matrix(deriv_sigmoid);

    }

    for(int k = index_last_layer; k >= 0 ; k--){
        if(k > 0 ) {
            if(n->layers[k].dw != NULL ) free_matrix(n->layers[k].dw);

            Matrix * temp = matrix_trans(n->layers[(k-1)%n->number_layers].output);
            n->layers[k].dw = matrix_product(n->layers[k].delta, temp);

            free_matrix(temp);
        }else{
            if(n->layers[k].dw != NULL ) free_matrix(n->layers[k].dw);

            Matrix * temp = matrix_trans(input);
            n->layers[k].dw = matrix_product(n->layers[k].delta,temp);

            free_matrix(temp);
        }

        if(n->layers[k].db != NULL ) free_matrix(n->layers[k].db);

        n->layers[k].db = matrix_copy(n->layers[k].delta);

    }

}

Matrix * apply_derivate_sigmoid(Matrix *m)
{
    if(m == NULL) return NULL;

    Matrix * new_matrix = create_matrix(m->rows,m->cols);
    if(new_matrix == NULL) return NULL;


    for(int i = 0; i < new_matrix->rows; i++){
        for(int j = 0; j <new_matrix->cols; j++){
            int index = i*new_matrix->cols+j;
            new_matrix->data[index] = derivate_sigmoid(m->data[index]);
        }
    }
    return new_matrix;
}


void network_update_weights(Network * n, double l_rate)
{
    for(int i = 0; i < n->number_layers; i++){
        n->layers[i].weights = network_update_B_W(n->layers[i].dw,n->layers[i].weights,l_rate);
        n->layers[i].bias = network_update_B_W(n->layers[i].db,n->layers[i].bias,l_rate);
    }
}

Matrix * network_update_B_W(Matrix * dw, Matrix * temp,  double l_rate)
{
    Matrix * new_dw = matrix_scalar_multiply(dw, l_rate);
    Matrix * new_matrix = matrix_sub(temp, new_dw);

    if(temp != NULL) free_matrix(temp);
    free_matrix(new_dw);

    return new_matrix;
}

void network_train_step(Network * n, Matrix * input, Matrix * goal, double l_rate)
{
    network_predict(n,input);
    network_backward(n,goal,input);
    network_update_weights(n,l_rate);
}


void network_save(Network * n, const char * path)
{

    FILE * file = fopen(path,"wb");
    if(file == NULL){
        perror("Error opening file1");
        return;
    }
    fwrite(&n->number_layers,sizeof(int),1,file);

    for(int i = 0 ; i < n->number_layers; i++){
        Layer * layer = &n->layers[i];
        fwrite(&layer->weights->rows,sizeof(int),1,file);
        fwrite(&layer->weights->cols,sizeof(int),1,file);
        fwrite(layer->weights->data,sizeof(double),layer->weights->rows*layer->weights->cols,file);

        fwrite(&layer->bias->rows,sizeof(int),1,file);
        fwrite(&layer->bias->cols,sizeof(int),1,file);
        fwrite(layer->bias->data,sizeof(double),layer->bias->rows*layer->bias->cols,file);
    }

    fclose(file);
}

Network * network_load(const char * path)
{
    Network * n = malloc(sizeof(Network));
    FILE * file = fopen(path,"rb");
    if(file == NULL){
        perror("Error opening file2");
        free(n);
        return NULL;
    }

    fread(&n->number_layers,sizeof(int),1,file);
    n->layers = calloc(n->number_layers,sizeof(Layer));

    for(int i = 0 ; i < n->number_layers; i++){
        Layer * layer = &n->layers[i];

        int rows, cols;
        fread(&rows,sizeof(int),1,file);
        fread(&cols,sizeof(int),1,file);
        layer->weights = create_matrix(rows,cols);
        fread(layer->weights->data,sizeof(double),rows*cols,file);

        fread(&rows,sizeof(int),1,file);
        fread(&cols,sizeof(int),1,file);
        layer->bias = create_matrix(rows,cols);
        fread(layer->bias->data,sizeof(double),rows*cols,file);
    }

    fclose(file);
    return n;
}


void free_network(Network *n)
{
    for(int i = 0; i < n->number_layers; i++){
        free_matrix(n->layers[i].bias);
        free_matrix(n->layers[i].weights);
        if(n->layers[i].output != NULL) free_matrix(n->layers[i].output);
        if(n->layers[i].dw != NULL) free_matrix(n->layers[i].dw);
        if(n->layers[i].db != NULL) free_matrix(n->layers[i].db);
        if(n->layers[i].delta != NULL) free_matrix(n->layers[i].delta);
    }
    free(n->layers);
    free(n);
}
