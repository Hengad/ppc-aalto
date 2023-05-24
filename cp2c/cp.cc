#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    std::vector<double> normData(ny * nx);
    int vectorizedDataWidth = floor(nx / 4); // if for example, nx = 82 then 84-82 = 2 elements will be padded 0
    if(nx % 4 != 0)
    {
        vectorizedDataWidth++;
    }
    std::vector<double4_t> vectorizedData(ny * vectorizedDataWidth);
    int i_1 = 0;
    int i_2 = 0;
    for(int j = 0; j < ny; j++)
    {
        // Calculate mean for a row
        double mean = 0;
        for(int i = 0; i < nx; i++)
        {
            double val = data[i + j*nx];
            mean += val;
            normData[i + j*nx] = val;
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
            vectorizedData[i_1][i_2] = normData[i + j*nx];
            i_2++;
            if(i_2 == 4)
            {
                i_2 = 0;
                i_1++;
            }
        }
        if(i_2 != 0)
        {
            while(i_2 < 4)
            {
                vectorizedData[i_1][i_2] = 0;
                i_2++;
            }
            i_2 = 0;
            i_1++;
        }
    }

    // Calculate matrix multiplication with normalized X and X^T
    for(int j = 0; j < ny; j++)
    {
        for(int i = 0; i < ny; i++)
        {
            // Calculate pairwise correlations
            if (j <= i)
            {
                // Multiply rows with vector operations and sum into resultVec
                double4_t resultVec = { 0, 0, 0, 0 };
                for(int k = 0; k < vectorizedDataWidth; k++)
                {
                    resultVec += vectorizedData[k + i*vectorizedDataWidth] * vectorizedData[k + j*vectorizedDataWidth];
                }
                // Calculate the sum of the 4 elements of resultVec to achieve the final sum
                double sum = 0;
                for(int sum_i = 0; sum_i < 4; sum_i++)
                {
                    sum += resultVec[sum_i];
                }
                result[i + j*ny] = sum;
            }
        }
    }
}