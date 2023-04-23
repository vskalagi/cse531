#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#define INF 200
using namespace std;
using namespace std::chrono;



static __global__
void self_dependent(const int blockId, size_t pitch, const int n_nodes, int* const matrix_data,int bs) {
    extern __shared__ int shared_data_[];

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = bs * blockId + idy;
    const int v2 = bs * blockId + idx;

    int newPath;
    int index=idy*bs+idx;

    const int cellId = v1 * pitch + v2;
    if (v1 < n_nodes && v2 < n_nodes) {
        shared_data_[index] = matrix_data[cellId];
    } else {
        shared_data_[index] = INF;
    }

    __syncthreads();

    
    for (int u = 0; u < bs; ++u) {
        newPath = shared_data_[idy*bs+u] + shared_data_[u*bs+idx];

        __syncthreads();
        if (newPath < shared_data_[index]) {
            shared_data_[index] = newPath;
        }
        __syncthreads();
    }

    if (v1 < n_nodes && v2 < n_nodes) {
        matrix_data[cellId] = shared_data_[index];
    }
}

static __global__
void pivot_row_column(const int blockId, size_t pitch, const int n_nodes, int* const matrix_data, int bs) {
    if (blockIdx.x == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    int v1 = bs * blockId + idy;
    int v2 = bs * blockId + idx;
    extern __shared__ int shared_data_Base[];
    
    int cellId = v1 * pitch + v2;
    int index=idy*bs+idx;

    if (v1 < n_nodes && v2 < n_nodes) {
        shared_data_Base[index] = matrix_data[cellId];
    } else {
        shared_data_Base[index] = INF;
    }

    if (blockIdx.y == 0) {
        v2 = bs * blockIdx.x + idx;
    } else {
        v1 = bs * blockIdx.x + idy;
    }

    int *shared_data_ = bs*bs + shared_data_Base;
    int currentPath;

    cellId = v1 * pitch + v2;
    if (v1 < n_nodes && v2 < n_nodes) {
        currentPath = matrix_data[cellId];
    } else {
        currentPath = INF;
    }
    shared_data_[index] = currentPath;
    __syncthreads();

    int newPath;
    if (blockIdx.y == 0) {
        
        for (int u = 0; u < bs; ++u) {
            newPath = shared_data_Base[idy*bs+u] + shared_data_[u*bs+idx];

            if (newPath < currentPath) {
                currentPath = newPath;
            }
            __syncthreads();

            shared_data_[index] = currentPath;

            __syncthreads();
        }
    } else {
        
        for (int u = 0; u < bs; ++u) {
            newPath = shared_data_[idy*bs+u] + shared_data_Base[u*bs+idx];

            if (newPath < currentPath) {
                currentPath = newPath;
            }

            __syncthreads();

            shared_data_[index] = currentPath;

            __syncthreads();
        }
    }

    if (v1 < n_nodes && v2 < n_nodes) {
        matrix_data[cellId] = currentPath;
    }
}

static __global__
void other_blocks(const int blockId, size_t pitch, const int n_nodes, int* const matrix_data,int bs) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;
    int index=idy*bs+idx;

    extern __shared__ int shared_data_BaseRow[];
    int *shared_data_BaseCol = bs*bs+shared_data_BaseRow;

    int v1Row = bs * blockId + idy;
    int v2Col = bs * blockId + idx;

    int cellId;
    if (v1Row < n_nodes && v2 < n_nodes) {
        cellId = v1Row * pitch + v2;

        shared_data_BaseRow[index] = matrix_data[cellId];
    }
    else {
        shared_data_BaseRow[index] = INF;
    }

    if (v1  < n_nodes && v2Col < n_nodes) {
        cellId = v1 * pitch + v2Col;
        shared_data_BaseCol[index] = matrix_data[cellId];
    }
    else {
        shared_data_BaseCol[index] = INF;
    }

   __syncthreads();

   int currentPath;
   int newPath;

   if (v1  < n_nodes && v2 < n_nodes) {
       cellId = v1 * pitch + v2;
       currentPath = matrix_data[cellId];

       
       for (int u = 0; u < bs; ++u) {
           newPath = shared_data_BaseCol[idy*bs+u] + shared_data_BaseRow[u*bs+idx];
           if (currentPath > newPath) {
               currentPath = newPath;
           }
       }
       matrix_data[cellId] = currentPath;
   }
}

void cudaBlockedFW(int *cpu_data,int n_nodes,int bs) {
    cudaSetDevice(0);
    //printf("%d-bb",bs);
    int *gpu_data;
    size_t pitch ;
    /* host to device */
    size_t columns=n_nodes * sizeof(int);
    cudaMallocPitch(&gpu_data, &pitch, columns, (size_t)n_nodes);
    cudaMemcpy2D(gpu_data, pitch,cpu_data, columns, columns, (size_t)n_nodes, cudaMemcpyHostToDevice);

    int n_block = (n_nodes - 1) / bs + 1;

    dim3 phase1(1 ,1, 1);
    dim3 phase2(n_block, 2 , 1);
    dim3 phase3(n_block, n_block , 1);
    dim3 dimBlockSize(bs, bs, 1);
    unsigned shared_mem_size_dependent = (bs*bs*sizeof(int));
    unsigned shared_mem_size_partial = (2*bs*bs*sizeof(int));	

    

    for(int blockID = 0; blockID < n_block; ++blockID) {
        self_dependent<<<phase1, dimBlockSize,shared_mem_size_dependent>>>
                (blockID, pitch / sizeof(int), n_nodes, gpu_data,bs);

        pivot_row_column<<<phase2, dimBlockSize, shared_mem_size_partial>>>
                (blockID, pitch / sizeof(int), n_nodes, gpu_data,bs);

        other_blocks<<<phase3, dimBlockSize,shared_mem_size_partial>>>
                (blockID, pitch / sizeof(int), n_nodes, gpu_data,bs);
    }

    cudaGetLastError();
    cudaDeviceSynchronize();
    /* device to host */
    cudaMemcpy2D(cpu_data, columns, gpu_data, pitch, columns, (size_t)n_nodes, cudaMemcpyDeviceToHost);
    cudaFree(gpu_data);
}


int main(int argc, char** argv) {
    int n, m, *d;
    // input
    FILE *infile = fopen(argv[1], "r");
    fscanf(infile, "%d %d", &n, &m);
    d = (int *) malloc(sizeof(int *) * n * n);
    for (int i = 0; i < n * n; ++i) d[i] = INF;
    int a, b, w;
    for (int i = 0; i < m; ++i) {
        fscanf(infile, "%d %d %d", &a, &b, &w);
        d[a * n + b] = d[b * n + a] = w;
    }
    fclose(infile);
    //printf("%s",argv[3]);
    int bs=atoi(argv[3]);
    auto start= high_resolution_clock::now();
    cudaBlockedFW(d,n,bs);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "Time taken by function: "<< duration.count() << " microseconds" << endl;
    // ouput
    FILE *outfile = fopen(argv[2], "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            fprintf(outfile, "%d%s",
                (i == j ? 0 : d[i * n + j]),
                (j == n - 1 ? " \n" : " ")
            );
        }
    }
    free(d);
}
