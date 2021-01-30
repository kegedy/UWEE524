#include <stdio.h>

// TESTS
// Test 1: Matrix additon. [3] + [4] = [7]

// Test 2: Dot Product. <2,2,2> * <3,3,3> = 6 * N

// Test 3: BLAS2. y = aMv + bw
//         where b = 4, w = <1>
//               a = 2, M =      


void initArr(float num, float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = num;
    }
}

void initMat(float num, float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = num;
    }
}

void checkElementsArr(float target, float* vector, int N) {
    for (int i = 0; i < N; i++) {
        if (vector[i] != target) {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
        }
    }
    printf("SUCCESS! All values calculated correctly.\n");
}

void checkElementsMat(float target, float* vector, int N) {
    for (int i = 0; i < N; i++) {
        if (vector[i] != target) {
            printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
        }
    }
    printf("SUCCESS! All values calculated correctly.\n");
}