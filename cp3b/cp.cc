#include <iostream>
#include <vector>
#include <cmath>

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<float> normData(ny * nx);
    int w = floor(nx / 8);
    int vectorizedDataWidth = (nx % 8 != 0) ? w + 1 : w; // just enough width for the data to fit in
    std::vector<float8_t> vectorizedData(ny * vectorizedDataWidth);
    int i_1 = 0;
    int i_2 = 0;
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
            vectorizedData[i_1][i_2] = normData[i + j*nx];
            // Keep the i_1 and i_2 indices up to date
            i_2++;
            if(i_2 == 8)
            {
                i_2 = 0;
                i_1++;
            }
        }
        // Zero padding to the vectorized data
        if(i_2 != 0)
        {
            while(i_2 < 8)
            {
                vectorizedData[i_1][i_2] = 0;
                i_2++;
            }
            i_2 = 0;
            i_1++;
        }
    }

    // Since we are now block looping i and j to nc, implement data reuse for each if clause
    const int blockSize = 40;
    int nc = (ny + blockSize - 1) / blockSize;

    // Add padding
    for(int p = ny; p < blockSize*nc; p++)
    {
        for(int pi = 0; pi < vectorizedDataWidth; pi++)
        {
            float8_t padVec = { 0, 0, 0, 0, 0, 0, 0, 0};
            vectorizedData.push_back(padVec);
        }                    
    }

    // Calculate matrix multiplication with normalized X and X^T
    // Multithreading with omp
    #pragma omp parallel for schedule(dynamic,1)
    for(int j = 0; j < nc; j++)
    {
        for(int i = 0; i < nc; i++)
        {
            if (j <= i)
            {
                std::vector<float8_t> vv(blockSize * blockSize);
                std::vector<float8_t> registerData1(2 * blockSize);
                std::vector<float8_t> registerData2(2 * blockSize);
                std::vector<float8_t> registerData3(2 * blockSize);
                std::vector<float8_t> registerData4(2 * blockSize);
                // No instruction level parallelization
                if(vectorizedDataWidth < 4)
                {
                    for(int k = 0; k < vectorizedDataWidth; k++)
                    {
                        // Initialize variables to keep in register
                        for(int vi = 0; vi < blockSize; vi++)
                        {
                            registerData1[vi] = vectorizedData[vectorizedDataWidth*(j * blockSize + vi) + k];
                            registerData1[vi + blockSize] = vectorizedData[vectorizedDataWidth*(i * blockSize + vi) + k];
                        }
                        // Calculate pairwise correlations with registerData1
                        for(int vj = 0; vj < blockSize; vj++)
                        {
                            for(int vi = 0; vi < blockSize; vi++)
                            {
                                vv[vi + vj*blockSize] += registerData1[vj] * registerData1[vi + blockSize];
                            }
                        }
                    }
                    // Loop through the block
                    for(int jd = 0; jd < blockSize; jd++)
                    {
                        for(int id = 0; id < blockSize; id++)
                        {
                            int real_i = i * blockSize + id;
                            int real_j = j * blockSize + jd;
                            if (real_i < ny && real_j < ny)
                            {
                                float sum = 0;
                                for(int vec_i = 0; vec_i < 8; vec_i++) // 8 = size of float8_t
                                {
                                    sum += vv[blockSize*jd + id][vec_i];
                                }
                                result[ny*real_j + real_i] = sum;
                            }
                        }
                    }
                }
                // Add instruction level parallelization when vectorizedDataWidth >= 4
                else
                {
                    int k = 0;
                    for(; k < vectorizedDataWidth; k+=4)
                    {
                        if (k < vectorizedDataWidth - (vectorizedDataWidth % 8))
                        {
                            // Initialize variables to keep in register
                            for(int vi = 0; vi < blockSize; vi++)
                            {
                                registerData1[vi] = vectorizedData[vectorizedDataWidth*(j * blockSize + vi) + k];
                                registerData1[vi + blockSize] = vectorizedData[vectorizedDataWidth*(i * blockSize + vi) + k];
                                
                                registerData2[vi] = vectorizedData[vectorizedDataWidth*(j * blockSize + vi) + k+1];
                                registerData2[vi + blockSize] = vectorizedData[vectorizedDataWidth*(i * blockSize + vi) + k+1];

                                registerData3[vi] = vectorizedData[vectorizedDataWidth*(j * blockSize + vi) + k+2];
                                registerData3[vi + blockSize] = vectorizedData[vectorizedDataWidth*(i * blockSize + vi) + k+2];

                                registerData4[vi] = vectorizedData[vectorizedDataWidth*(j * blockSize + vi) + k+3];
                                registerData4[vi + blockSize] = vectorizedData[vectorizedDataWidth*(i * blockSize + vi) + k+3];
                            }
                            // Calculate pairwise correlations
                            for(int vj = 0; vj < blockSize; vj++)
                            {
                                for(int vi = 0; vi < blockSize; vi++)
                                {
                                    float8_t a = registerData1[vj] * registerData1[vi + blockSize];
                                    float8_t b = registerData2[vj] * registerData2[vi + blockSize];
                                    float8_t c = registerData3[vj] * registerData3[vi + blockSize];
                                    float8_t d = registerData4[vj] * registerData4[vi + blockSize];
                                    vv[vi + vj*blockSize] += a + b + c + d;
                                }
                            }
                        }
                        // If nx is not divisible by 8, deal with the leftover rows
                        else
                        {
                            for(; k < vectorizedDataWidth; k++)
                            {
                                // Initialize variables to keep in register
                                for(int vi = 0; vi < blockSize; vi++)
                                {
                                    registerData1[vi] = vectorizedData[vectorizedDataWidth*(j * blockSize + vi) + k];
                                    registerData1[vi + blockSize] = vectorizedData[vectorizedDataWidth*(i * blockSize + vi) + k];
                                }
                                // Calculate pairwise correlations with registerData1
                                for(int vj = 0; vj < blockSize; vj++)
                                {
                                    for(int vi = 0; vi < blockSize; vi++)
                                    {
                                        vv[vi + vj*blockSize] += registerData1[vj] * registerData1[vi + blockSize];
                                    }
                                }
                            }
                        }
                    }
                }
                // Loop through the block
                for(int jd = 0; jd < blockSize; jd++)
                {
                    for(int id = 0; id < blockSize; id++)
                    {
                        int real_i = i * blockSize + id;
                        int real_j = j * blockSize + jd;
                        if (real_i < ny && real_j < ny)
                        {
                            float sum = 0;
                            for(int vec_i = 0; vec_i < 8; vec_i++) // 8 = size of float8_t
                            {
                                sum += vv[blockSize*jd + id][vec_i];
                            }
                            result[ny*real_j + real_i] = sum;
                        }
                    }
                }
            }
        }
    }
}
