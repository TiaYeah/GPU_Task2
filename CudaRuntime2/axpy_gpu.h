#pragma once

void saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, int block_size);

void daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, int block_size);
