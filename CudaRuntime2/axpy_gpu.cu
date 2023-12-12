#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

#include <stdio.h>
#include <omp.h>
#include "axpy_gpu.h"

__global__ void saxpy_gpuKernel(int n, float a, float* x, int incx, float* y, int incy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

__global__ void daxpy_gpuKernel(int n, double a, double* x, int incx, double* y, int incy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i * incy] = y[i * incy] + a * x[i * incx];
    }
}

void saxpy_gpu(int n, float a, float* x, int incx, float* y, int incy, int block_size) {
    float* x_gpu;
    float* y_gpu;
    cudaError_t cudaStatus;
    
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }
    
    // Allocate GPU buffers for three vectors (two input, one output)    .
    
    int x_gpu_size = 1 + (n - 1) * incx;
    int y_gpu_size = 1 + (n - 1) * incy;

    cudaStatus = cudaMalloc((void**)&x_gpu, x_gpu_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&y_gpu, y_gpu_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(x_gpu, x, x_gpu_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    cudaStatus = cudaMemcpy(y_gpu, y, y_gpu_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
    // Launch a kernel on the GPU with one thread for each element.
    //const int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    double begin, end;

    begin = omp_get_wtime();
    saxpy_gpuKernel <<<num_blocks, block_size >>> (n, a, x_gpu, incx, y_gpu, incy);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "saxpy_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    end = omp_get_wtime();

    printf("GPU time with block size %d: %f\n", block_size, end - begin);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching saxpy_gpu!\n", cudaStatus);
        goto Error;
    }
    
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(y, y_gpu, y_gpu_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    
Error:
    cudaFree(x_gpu);
    cudaFree(y_gpu);
        
}

void daxpy_gpu(int n, double a, double* x, int incx, double* y, int incy, int block_size) {
    double* x_gpu;
    double* y_gpu;
    cudaError_t cudaStatus;


    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .

    int x_gpu_size = 1 + (n - 1) * incx;
    int y_gpu_size = 1 + (n - 1) * incy;

    cudaStatus = cudaMalloc((void**)&x_gpu, x_gpu_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&y_gpu, y_gpu_size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(x_gpu, x, x_gpu_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(y_gpu, y, y_gpu_size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    int num_blocks = (n + block_size - 1) / block_size;
    double begin, end;

    begin = omp_get_wtime();
    daxpy_gpuKernel << <num_blocks, block_size >> > (n, a, x_gpu, incx, y_gpu, incy);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "daxpy_gpu launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    end = omp_get_wtime();

    printf("GPU time with block size %d: %f\n", block_size, end - begin);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching daxpy_gpu!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(y, y_gpu, y_gpu_size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(x_gpu);
    cudaFree(y_gpu);

}
