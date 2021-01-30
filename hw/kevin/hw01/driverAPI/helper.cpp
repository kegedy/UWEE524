#include <stdio.h>
#include <assert.h>

// https://gist.github.com/tautologico/2879581
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(0);
    }
}

// TIME OPERATIONS: 
// https://stackoverflow.com/questions/1739259/how-to-use-queryperformancecounter
#define TIMER_INIT \
    LARGE_INTEGER frequency; \
    LARGE_INTEGER t1,t2; \
    double elapsedTime; \
    QueryPerformanceFrequency(&frequency);

#define TIMER_START QueryPerformanceCounter(&t1);

#define TIMER_STOP \
    QueryPerformanceCounter(&t2); \
    elapsedTime=(float)(t2.QuadPart-t1.QuadPart)/frequency.QuadPart; \
    std::wcout<<elapsedTime<<L" sec"<<endl;

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