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
                // Calculate correlation coefficient
                double sum = 0;
                for(int k = 0; k < nx; k++)
                {
                    sum += normData[k+i*nx] * normData[k+j*nx];
                }
                result[i +j*ny] = sum;
            }
        }
    }
}