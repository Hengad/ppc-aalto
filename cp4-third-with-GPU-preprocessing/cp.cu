/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

#define CHECK(x) check(x, #x)


__global__ void preProcessKernel(float *r, const float *d, int ny, int nx)
{
    /*
    //std::vector<float> normData(ny * nx);
    for(int j = 0; j < ny; j++)
    {
        // Calculate mean for a row
        float mean = 0;
        for(int i = 0; i < nx; i++)
        {
            float val = data[i + j*nx];
            mean += val;
            normData[i + j*nx] = val;
        }
        mean /= nx;

        // Calculate euclidean length for a row after it's mean normalized
        float length = 0;
        for(int i = 0; i < nx; i++)
        {
            float val = normData[i + j*nx] - mean;
            normData[i + j*nx] = val;
            length += pow(val, 2);
        }
        length = sqrt(length);
        
        // Euclidean length normalization
        for(int i = 0; i < nx; i++)
        {
            normData[i + j*nx] /= length;
        }
    }
    */
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= ny || j >= ny)
        return;
}

__global__ void coefficientKernel(float *r, const float *d, int ny, int nx)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if(i >= ny || j >= ny)
        return;
    if (j <= i)
    {
        // Calculate correlation coefficient
        float sum = 0;
        for(int k = 0; k < nx; k++)
        {
            sum += d[k+j*nx] * d[k+i*nx];
        }
        r[i+j*ny] = sum;
    }
    else
    {
        r[i+j*ny] = 0;
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    // Allocate memory and copy data to GPU
    float *dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, ny * nx * sizeof(float)));
    float *normDataGPU = NULL;
    CHECK(cudaMalloc((void**)&normDataGPU, ny * nx * sizeof(float)));
    float *resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, normData.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Setup blocks and threads
    dim3 dimBlock(8, 8);
    dim3 dimGrid1(divup(ny, dimBlock.x), divup(nx, dimBlock.y));
    dim3 dimGrid2(divup(ny, dimBlock.x), divup(ny, dimBlock.y));

    // Run kernel for data preprocessing
    preProcessKernel<<<dimGrid1, dimBlock>>>(normDataGPU, dataGPU, ny, nx);

    // Run kernel for coefficient calculations
    coefficientKernel<<<dimGrid2, dimBlock>>>(resultGPU, normDataGPU, ny, nx);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(normDataGPU));
    CHECK(cudaFree(resultGPU));
}
