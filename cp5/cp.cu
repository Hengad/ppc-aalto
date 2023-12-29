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

#define CHECK(x) check(x, #x)

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

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

__global__ void myppkernel(float *r, const float *d, int ny, int nx, int nn_y, int nn_x)
{
    int ja = threadIdx.x;
    int i = blockIdx.y;

    for(int jb = 0; jb < nn_x; jb += 64)
    {
        int j = jb + ja;
        float v = (i < ny && j < nx) ? d[nx * i + j] : 0;
        r[nn_y * j + i] = v;
    }
}

__global__ void coefficientKernel(float *r, const float *d, int ny, int nx, int nn)
{
    //int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int j = threadIdx.y + blockIdx.y * blockDim.y;
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;


    if(ic <= jc)
    {
        float v[8][8];
        for(int ib = 0; ib < 8; ib++)
        {
            for(int jb = 0; jb < 8; jb++)
            {
                v[ib][jb] = 0;
            }
        }
        for(int k = 0; k < nx; k ++)
        {
            float x[8];
            float y[8];
            for(int ib = 0; ib < 8; ib++)
            {
                int i = ic * 64 + ib * 8 + ia;
                //x[ib] = d[k + i*nx];
                x[ib] = d[nn * k + i];
            }
            for(int jb = 0; jb < 8; jb++)
            {
                int j = jc * 64 + jb * 8 + ja;
                y[jb] = d[nn * k + j];
            }
            for(int ib = 0; ib < 8; ib++)
            {
                for(int jb = 0; jb < 8; jb++)
                {
                    v[ib][jb] += x[ib] * y[jb];
                }
            }
        }
        for(int ib = 0; ib < 8; ib++)
        {
            for(int jb = 0; jb < 8; jb++)
            {
                int i = ic * 64 + ib * 8 + ia;
                int j = jc * 64 + jb * 8 + ja;
                if(i < ny && j < ny)
                    r[j + i*ny] = v[ib][jb];
            }
        }
    }
    else
    {
        for(int ib = 0; ib < 8; ib++)
        {
            for(int jb = 0; jb < 8; jb++)
            {
                int i = ic * 64 + ib * 8 + ia;
                int j = jc * 64 + jb * 8 + ja;
                if(i < ny && j < ny)
                    r[j + i*ny] = 0;
            }
        }
    }
}

void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<float> normData(ny * nx);
    for(int j = 0; j < ny; j++)
    {
        // Calculate mean for a row
        float mean = 0;
        for(int i = 0; i < nx; i++)
        {
            float val = data[i + j*nx];
            mean += val;
        }
        mean /= nx;

        // Calculate euclidean length for a row after it's mean normalized
        float length = 0;
        for(int i = 0; i < nx; i++)
        {
            float val = data[i + j*nx] - mean;
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

    int nn_y = roundup(ny, 64);
    int nn_x = roundup(nx, 64);

    // Allocate memory & copy data to GPU 
    float *dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, nn_y * nn_x * sizeof(float)));
    float *dataNormGPU = NULL;
    CHECK(cudaMalloc((void**)&dataNormGPU, ny * nx * sizeof(float)));
    float *resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dataNormGPU, normData.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Add padding
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn_y);
        myppkernel<<<dimGrid, dimBlock>>>(dataGPU, dataNormGPU, ny, nx, nn_y, nn_x);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    }

    // Run kernel for coefficient calculations
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn_y / 64, nn_y / 64);
        coefficientKernel<<<dimGrid, dimBlock>>>(resultGPU, dataGPU, ny, nx, nn_y);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(dataNormGPU));
    CHECK(cudaFree(resultGPU));
}
