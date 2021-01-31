#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


void initArr(float num, float* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = num;
    }
}

void initMat(float num, float** a, int M, int N) {
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            a[i][j] = num;
        }
    }
}

void checkElementsArr(float target, float* arr, int N) {
    for (int i = 0; i < N; i++) {
        if (arr[i] != target) {
            printf("FAIL: arr[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
        }
    }
    printf("SUCCESS! All values calculated correctly.\n");
}

void checkElementsMat(float target, float** a, int M, int N) {
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            if (a[i][j] != target) {
                printf("FAIL: a[%d][%d] - %0.0f does not equal %0.0f\n", i, j, vector[i], target);
            }
        }
    }
    printf("SUCCESS! All values calculated correctly.\n");
}

float sumArr(float* arr, int N) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += arr[i];
    }
    return sum;
}

float dotproduct(float* a, float* b, int N) {
    float sum = 0;
    for (int i = 0; i < N; i++) {
        sum += a[i]*b[i];
    }
    return sum;
}