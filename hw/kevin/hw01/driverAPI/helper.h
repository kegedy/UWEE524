#ifndef HELPER_H
#define HELPER_H

void initArr(float num, float* a, int N);
void initMat(float target, float** a, int M, int N);
void checkElementsArr(float target, float* vector, int N);
void checkElementsMat(float target, float** a, int M, int N);
float sumArr(float* arr, int N);
float dotproduct(float* a, float* b, int N);

#endif HELPER_H