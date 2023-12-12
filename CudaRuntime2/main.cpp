#include <cstdio>
#include <iostream>
#include <vector>
#include <cassert>

#include "axpy.h"
#include "axpy_gpu.h"
#include "axpy_omp.h"


using namespace std;

const int n = 1 << 24;

bool compareArrays(float* y, float* res, int size){
    for (int i = 0; i < size; i++) {
        if (y[i] != res[i]) {
            return false;
        }
    }
    return true;
}

bool compareArrays(double* y, double* res, int size) {
    for (int i = 0; i < size; i++) {
        if (y[i] != res[i]) {
            return false;
        }
    }
    return true;
}

void printArray(float* y, int size) {
    for (int i = 0; i < size; i++) {
        cout << y[i] << " ";
    }
    cout << endl;
}

void printArray(double* y, int size) {
    for (int i = 0; i < size; i++) {
        cout << y[i] << " ";
    }
    cout << endl;
}

void clear_y(float* y, int size_y) {
    for (int i = 0; i < size_y; i++) {
        y[i] = 0.0f;
    }
}

void clear_y(double* y, int size_y) {
    for (int i = 0; i < size_y; i++) {
        y[i] = 0.0f;
    }
}

void test(int n, float a, float* x, int incx, float* y, int incy, float* res) {
    std::vector<int> block_sizes = { 8, 16, 32, 64, 128, 256 };
    int size_y = 1 + (n - 1) * incy;

    saxpy(n, a, x, incx, y, incy);
    assert(compareArrays(y, res, size_y));
    clear_y(y, size_y);

    saxpy_omp(n, a, x, incx, y, incy);
    assert(compareArrays(y, res, size_y));
    clear_y(y, size_y);

    for (int i = 0; i < block_sizes.size(); i++) {
        saxpy_gpu(n, a, x, incx, y, incy, block_sizes[i]);
        assert(compareArrays(y, res, size_y));
        clear_y(y, size_y);
    }
}

void test(int n, double a, double* x, int incx, double* y, int incy, double* res) {
    std::vector<int> block_sizes = { 8, 16, 32, 64, 128, 256 };
    int size_y = 1 + (n - 1) * incy;

    daxpy(n, a, x, incx, y, incy);
    assert(compareArrays(y, res, size_y));
    clear_y(y, size_y);

    daxpy_omp(n, a, x, incx, y, incy);
    assert(compareArrays(y, res, size_y));
    clear_y(y, size_y);

    for (int i = 0; i < block_sizes.size(); i++) {
        daxpy_gpu(n, a, x, incx, y, incy, block_sizes[i]);
        assert(compareArrays(y, res, size_y));
        clear_y(y, size_y);
    }
}

void testFloat() {
    float a = 1.2f;
    int incx = 3, incy = 2;
    float* x, * y, * res;

    int size_x = 1 + (n - 1) * incx;
    int size_y = 1 + (n - 1) * incy;

    x = new float[size_x];
    y = new float[size_y];
    res = new float[size_y];

    for (int i = 0; i < size_x; i++) {
        x[i] = 0.0f;
    }
    for (int i = 0; i < size_x; i += incx) {
        x[i] = 1.0f;
    }

    for (int i = 0; i < size_y; i++) {
        y[i] = 0.0f;
        if (i % 2 == 0) {
            res[i] = 1.2f;
        }
        else {
            res[i] = 0.0f;
        }
    }

    cout << "FLOAT TEST 1\n";
    test(n, a, x, incx, y, incy, res);
    delete[] x, y, res;


    a = 5.0f;
    incy = 4;
    incx = 1;
    size_x = 1 + (n - 1) * incx;
    size_y = 1 + (n - 1) * incy;

    x = new float[size_x];
    y = new float[size_y];
    res = new float[size_y];
    for (int i = 0; i < size_x; i++) {
        x[i] = 0.0f;
    }
    for (int i = 0; i < size_x; i += incx) {
        x[i] = 1.5f;
    }

    for (int i = 0; i < size_y; i++) {
        y[i] = 0.0f;
        if (i % 4 == 0) {
            res[i] = 7.5f;
        }
        else {
            res[i] = 0.0f;
        }
    }

    cout << "FLOAT TEST 2\n";
    test(n, a, x, incx, y, incy, res);
}

void testDouble() {
    double a = 1.2;
    int incx = 3, incy = 2;
    double* x, * y, * res;

    int size_x = 1 + (n - 1) * incx;
    int size_y = 1 + (n - 1) * incy;

    x = new double[size_x];
    y = new double[size_y];
    res = new double[size_y];

    for (int i = 0; i < size_x; i++) {
        x[i] = 0.0;
    }
    for (int i = 0; i < size_x; i += incx) {
        x[i] = 1.0;
    }

    for (int i = 0; i < size_y; i++) {
        y[i] = 0.0;
        if (i % 2 == 0) {
            res[i] = 1.2;
        }
        else {
            res[i] = 0.0;
        }
    }

    cout << "DOUBLE TEST 1\n";
    test(n, a, x, incx, y, incy, res);
    delete[] x, y, res;

    a = 5.0;
    incy = 4;
    incx = 1;
    size_x = 1 + (n - 1) * incx;
    size_y = 1 + (n - 1) * incy;

    x = new double[size_x];
    y = new double[size_y];
    res = new double[size_y];
    for (int i = 0; i < size_x; i++) {
        x[i] = 0.0;
    }
    for (int i = 0; i < size_x; i += incx) {
        x[i] = 1.5;
    }

    for (int i = 0; i < size_y; i++) {
        y[i] = 0.0;
        if (i % 4 == 0) {
            res[i] = 7.5;
        }
        else {
            res[i] = 0.0;
        }
    }

    cout << "DOUBLE TEST 2\n";
    test(n, a, x, incx, y, incy, res);
}

int main() {
    testFloat();
    testDouble();
    

    return 0;
}


