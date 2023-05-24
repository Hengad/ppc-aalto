#include <iostream>
#include <vector>
#include <cmath>

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<double> normData;
    normData.reserve(ny * nx);
    for(int j = 0; j < ny; j++)
    {
        // Calculate mean for a row
        double mean = 0;
        for(int i = 0; i < nx; i++)
        {
            double val = data[i + j*nx];
            mean += val;
            normData.push_back(val);
        }
        mean /= nx;

        // Calculate euclidean length for a row after it's mean normalized
        double length = 0;
        for(int i = 0; i < nx; i++)
        {
            double val = normData[i + j*nx] - mean;
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

    // Calculate matrix multiplication with normalized X and X^T
    for(int j = 0; j < ny; j++)
    {
        for(int i = 0; i < ny; i++)
        {
            if (j <= i)
            {
                double sum = 0;
                double a = 0;
                double b = 0;
                double c = 0;
                double d = 0;
                // Can't be instruction level parallelized
                if (nx == 1)
                {
                    for(int k = 0; k < nx; k++)
                    {
                        sum += normData[k+i*nx] * normData[k + j*nx];
                    }
                }
                // Instruction level parallelization with nx == 2
                else if(nx == 2)
                {
                    for(int k = 0; k < nx; k+=2)
                    {
                        a += normData[k+i*nx] * normData[k+j*nx];
                        b += normData[k+i*nx+1] * normData[k+j*nx+1];
                    }
                }
                // Instruction level parallelization with nx == 3
                else if(nx == 3)
                {
                    for(int k = 0; k < nx; k+=3)
                    {
                        a += normData[k+i*nx] * normData[k+j*nx];
                        b += normData[k+i*nx+1] * normData[k+j*nx+1];
                        b += normData[k+i*nx+2] * normData[k+j*nx+2];
                    }
                }
                // Instruction level parallelization with nx >= 4
                else if(nx >= 4)
                {
                    int k = 0;
                    for(; k < nx; k+=4)
                    {
                        if (k < nx - (nx % 4))
                        {
                            a += normData[k+i*nx] * normData[k+j*nx];
                            b += normData[k+i*nx+1] * normData[k+j*nx+1];
                            c += normData[k+i*nx+2] * normData[k+j*nx+2];
                            d += normData[k+i*nx+3] * normData[k+j*nx+3];
                        }
                        // If nx is not divisible by 4, deal with the leftover rows
                        else
                        {
                            for(; k < nx; k++)
                            {
                                a += normData[k+i*nx] * normData[k+j*nx];
                            }
                        }
                    }
                }
                
                sum = a + b + c + d;
                result[i +j*ny] = sum;
            }
        }
    }
}