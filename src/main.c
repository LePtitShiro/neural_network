#include "function_matrix.h"
#include <time.h>
#include <stdlib.h>


int main(){
    srand(time(NULL));

    Matrix * A = create_matrix(2, 3);
    Matrix * B = create_matrix(3, 2);

    // Remplissage de la matrice A (2x3)
    // [ 1, 2, 3 ]
    // [ 4, 5, 6 ]
    A->data[0] = 1; A->data[1] = 2; A->data[2] = 3;
    A->data[3] = 4; A->data[4] = 5; A->data[5] = 6;
    print_matrix(A);
    // Remplissage de la matrice B (3x2)
    // [ 7,  8 ]
    // [ 9,  10]
    // [ 11, 12]
    B->data[0] = 7;  B->data[1] = 8;
    B->data[2] = 9;  B->data[3] = 10;
    B->data[4] = 11; B->data[5] = 12;
    print_matrix(B);

    Matrix * C = matrix_product(A, B);

    print_matrix(C);

    Matrix * D = create_matrix(2,2);
    D->data[0] = 1; D->data[1] = 1;
    D->data[2] = 1; D->data[3] = 1;

    Matrix * E = matrix_add(C, D);
    print_matrix(E);


    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(D);
    free_matrix(E);
}
