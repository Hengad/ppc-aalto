#include <iostream>
#include <algorithm>
#include <vector>
/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{
    for(int j = 0; j < ny; j++)
    {
        for(int i = 0; i < nx; i++)
        {
            std::vector<double> window;
            window.reserve((2*hx+1)*(2*hy+1));
            double median;
            int length_x = std::min(i+hx+1, nx) - std::max(i-hx, 0);
            int length_y = std::min(j+hy+1, ny) - std::max(j-hy, 0);
            int length = length_x * length_y;
            for(int wy = std::max(j-hy, 0); wy <= std::min(j+hy, ny-1); wy++)
            {
                for(int wx = std::max(i-hx, 0); wx <= std::min(i+hx, nx-1); wx++)
                {
                    window.push_back(in[wx + nx*wy]);
                }
            }
            if(length % 2 != 0)
            {
                // n = 2k + 1
                std::nth_element(   window.begin(),
                                    window.begin() + ((length-1)/2)+1-1,
                                    window.end()
                                );
                median = window[((length-1)/2)+1-1];
            }
            else
            {
                // n = 2k
                std::nth_element(   window.begin(),
                                    window.begin() + (length/2)+1-1,
                                    window.end()
                                );
                double a = window[(length/2)+1-1];

                std::nth_element(   window.begin(),
                                    window.begin() + length/2-1,
                                    window.end()
                                );

                double b = window[length/2-1];

                median = (a+b)/2;
            }
            out[i + nx*j] = median;
        } 
    }
}