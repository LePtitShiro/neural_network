#include "function_matrix.h"
#include "network.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>


int main(){

    srand(time(NULL));

    Network * n = create_network(3,(int[]){2,4,1});

    Matrix * input[4];
    Matrix * goal[4];

    input[0] = create_matrix(2,1); input[0]->data[0] = 0; input[0]->data[1] = 0;
    input[1] = create_matrix(2,1); input[1]->data[0] = 0; input[1]->data[1] = 1;
    input[2] = create_matrix(2,1); input[2]->data[0] = 1; input[2]->data[1] = 0;
    input[3] = create_matrix(2,1); input[3]->data[0] = 1; input[3]->data[1] = 1;

    goal[0] = create_matrix(1,1); goal[0]->data[0] = 0;
    goal[1] = create_matrix(1,1); goal[1]->data[0] = 1;
    goal[2] = create_matrix(1,1); goal[2]->data[0] = 1;
    goal[3] = create_matrix(1,1); goal[3]->data[0] = 0;

    printf("TRAINING\n");
    for(int i = 0; i < 1000000; i++){
        for(int j = 0; j < 4; j++){
            network_train_step(n, input[j], goal[j], 0.1);
        }
        if (i % 1000 == 0) {
            printf("%lf\n", mse_cost(n->layers[n->number_layers - 1].output,goal[3]));
        }
    }

    network_save(n,"xor_save.bin");
    free_network(n);
    Network * n2 = network_load("xor_save.bin");


    printf("PREDICT\n");
    for(int i = 0; i < 4; i++){
        network_predict(n2,input[i]);
        printf("%lf\n",n2->layers[n2->number_layers - 1].output->data[0]);
    }

    for(int i = 0; i < 4; i++){
        free_matrix(input[i]);
        free_matrix(goal[i]);
    }

    free_network(n2);

    return 0;


}
