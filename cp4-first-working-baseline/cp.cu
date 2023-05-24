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

#define CHECK(x) check(x, #x)

/*
__global__ void preProcessKernel(float *r, const float *d, int ny, int nx)
{

}
*/
__global__ void coefficientKernel(float *r, const float *d, int ny, int nx)
{
    /*
    // Calculate matrix multiplication with normalized X and X^T
    for(int j = 0; j < ny; j++)
    {
        for(int i = 0; i < ny; i++)
        {
            if (j <= i)
            {
                // Calculate correlation coefficient
                float sum = 0;
                for(int k = 0; k < nx; k++)
                {
                    sum += normData[k+i*nx] * normData[k+j*nx];
                }
                result[i +j*ny] = sum;
            }
        }
    }*/
    int i = threadIdx.x;
    int j = blockIdx.x;

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
    /*
    // Allocate memory & copy data to GPU 
    float *dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, ny * nx * sizeof(float)));
    float *resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    */
    std::vector<float> normData;
    normData.reserve(ny * nx);
    for(int j = 0; j < ny; j++)
    {
        // Calculate mean for a row
        float mean = 0;
        for(int i = 0; i < nx; i++)
        {
            float val = data[i + j*nx];
            mean += val;
            normData.push_back(val);
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

    // Allocate memory & copy data to GPU 
    float *dataGPU = NULL;
    CHECK(cudaMalloc((void**)&dataGPU, ny * nx * sizeof(float)));
    float *resultGPU = NULL;
    CHECK(cudaMalloc((void**)&resultGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dataGPU, normData.data(), ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    coefficientKernel<<<ny, ny>>>(resultGPU, dataGPU, ny, nx);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, resultGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dataGPU));
    CHECK(cudaFree(resultGPU));
}
