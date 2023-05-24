#include <omp.h>
#include <cmath>
#include <algorithm>
typedef unsigned long long data_t;

void parallelMergeSort(int t, int n, data_t *data) {
    if (t <= 0) {
        std::sort(data, data + n);
        return;
    }
    
    int m = (n + 1) / 2;
    
    #pragma omp task
    parallelMergeSort(t - 1, m, data);
    
    #pragma omp task
    parallelMergeSort(t - 1, n - m, data + m);
    
    #pragma omp taskwait
    {
        data_t* temp = new data_t[n];
        std::merge(data, data + m, data + m, data + n, temp);
        std::copy(temp, temp + n, data);
        delete[] temp;
    }
}

void psort(int n, data_t *data) {
    int t = static_cast<int>(std::log2(omp_get_max_threads())) * 2;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelMergeSort(t, n, data);
        }
    }
}