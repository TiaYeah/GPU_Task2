#pragma once
#include "omp.h"
#include <stdio.h>


void saxpy_omp(int n, float a, float* x, int incx, float* y, int incy) {
    double begin, end;

    begin = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
    end = omp_get_wtime();
    printf("OMP time: %f\n", end - begin);
}

void daxpy_omp(int n, double a, double* x, int incx, double* y, int incy) {
    double begin, end;

    begin = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
    end = omp_get_wtime();
    printf("OMP time: %f\n", end - begin);
}