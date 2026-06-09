#ifndef NETWORK
#define NETWORK

#include "function_matrix.h"

typedef struct{
    Matrix * weights;
    Matrix * bias;
    Matrix * output;
    Matrix * dw;
    Matrix * db;
    Matrix * delta;
}Layer;

typedef struct{
    int number_layers;
    Layer * layers;
}Network;


void init_weights(Matrix* m, int e, int s);

void init_bias(Matrix* m);

double random_double(double a,double b);

void network_predict(Network * n, Matrix * input);

Matrix * apply_sigmoid(Matrix * m);









#endif
