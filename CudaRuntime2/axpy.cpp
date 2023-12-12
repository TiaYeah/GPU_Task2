#include "axpy.h"
#include "omp.h"
#include "stdio.h"

void saxpy(int n, float a, float* x, int incx, float* y, int incy) {
    double begin, end;

    begin = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
    end = omp_get_wtime();
    printf("CPU time: %f\n", end - begin);
}

void daxpy(int n, double a, double* x, int incx, double* y, int incy) {
    double begin, end;

    begin = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
    end = omp_get_wtime();
    printf("CPU time: %f\n", end - begin);
}