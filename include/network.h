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

void network_backward(Network * n, Matrix * goal, Matrix * input);

Matrix * apply_sigmoid(Matrix * m);

Matrix * apply_derivate_sigmoid(Matrix * m);

void network_update_weights(Network * n, double l_rate);

Matrix * network_update_B_W(Matrix * dw, Matrix * temp,  double l_rate);

void network_train_step(Network * n, Matrix * input, Matrix * goal, double l_rate);





#endif
