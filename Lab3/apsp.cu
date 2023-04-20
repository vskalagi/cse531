#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#define INF 200
#define MAX_DISTANCE 1 << 30 - 1


static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph,int bs) {
    extern __shared__ int cacheGraph[];

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = bs * blockId + idy;
    const int v2 = bs * blockId + idx;

    int newPath;
    int index=idy*bs+idx;

    const int cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraph[index] = graph[cellId];
    } else {
        cacheGraph[index] = MAX_DISTANCE;
    }

    __syncthreads();

    #pragma unroll
    for (int u = 0; u < bs; ++u) {
        newPath = cacheGraph[idy*bs+u] + cacheGraph[u*bs+idx];

        __syncthreads();
        if (newPath < cacheGraph[index]) {
            cacheGraph[index] = newPath;
        }
        __syncthreads();
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = cacheGraph[index];
    }
}

static __global__
void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph, int bs) {
    if (blockIdx.x == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    int v1 = bs * blockId + idy;
    int v2 = bs * blockId + idx;
    extern __shared__ int cacheGraphBase[];
    
    int cellId = v1 * pitch + v2;
    int index=idy*bs+idx;

    if (v1 < nvertex && v2 < nvertex) {
        cacheGraphBase[index] = graph[cellId];
    } else {
        cacheGraphBase[index] = MAX_DISTANCE;
    }

    if (blockIdx.y == 0) {
        v2 = bs * blockIdx.x + idx;
    } else {
        v1 = bs * blockIdx.x + idy;
    }

    int *cacheGraph = bs*bs + cacheGraphBase;
    int currentPath;

    cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        currentPath = graph[cellId];
    } else {
        currentPath = MAX_DISTANCE;
    }
    cacheGraph[index] = currentPath;
    __syncthreads();

    int newPath;
    if (blockIdx.y == 0) {
        #pragma unroll
        for (int u = 0; u < bs; ++u) {
            newPath = cacheGraphBase[idy*bs+u] + cacheGraph[u*bs+idx];

            if (newPath < currentPath) {
                currentPath = newPath;
            }
            __syncthreads();

            cacheGraph[index] = currentPath;

            __syncthreads();
        }
    } else {
        #pragma unroll
        for (int u = 0; u < bs; ++u) {
            newPath = cacheGraph[idy*bs+u] + cacheGraphBase[u*bs+idx];

            if (newPath < currentPath) {
                currentPath = newPath;
            }

            __syncthreads();

            cacheGraph[index] = currentPath;

            __syncthreads();
        }
    }

    if (v1 < nvertex && v2 < nvertex) {
        graph[cellId] = currentPath;
    }
}

static __global__
void _blocked_fw_independent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph,int bs) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;

    const int idx = threadIdx.x;
    const int idy = threadIdx.y;

    const int v1 = blockDim.y * blockIdx.y + idy;
    const int v2 = blockDim.x * blockIdx.x + idx;
    int index=idy*bs+idx;

    extern __shared__ int cacheGraphBaseRow[];
    int *cacheGraphBaseCol = bs*bs+cacheGraphBaseRow;

    int v1Row = bs * blockId + idy;
    int v2Col = bs * blockId + idx;

    int cellId;
    if (v1Row < nvertex && v2 < nvertex) {
        cellId = v1Row * pitch + v2;

        cacheGraphBaseRow[index] = graph[cellId];
    }
    else {
        cacheGraphBaseRow[index] = MAX_DISTANCE;
    }

    if (v1  < nvertex && v2Col < nvertex) {
        cellId = v1 * pitch + v2Col;
        cacheGraphBaseCol[index] = graph[cellId];
    }
    else {
        cacheGraphBaseCol[index] = MAX_DISTANCE;
    }

   __syncthreads();

   int currentPath;
   int newPath;

   if (v1  < nvertex && v2 < nvertex) {
       cellId = v1 * pitch + v2;
       currentPath = graph[cellId];

        #pragma unroll
       for (int u = 0; u < bs; ++u) {
           newPath = cacheGraphBaseCol[idy*bs+u] + cacheGraphBaseRow[u*bs+idx];
           if (currentPath > newPath) {
               currentPath = newPath;
           }
       }
       graph[cellId] = currentPath;
   }
}

static
size_t _cudaMoveMemoryToDevice(const int*  dataHost, int **graphDevice, int nvertex) {
    size_t height = nvertex;
    size_t width = height * sizeof(int);
    size_t pitch;

    cudaMallocPitch(graphDevice, &pitch, width, height);

    cudaMemcpy2D(*graphDevice, pitch,dataHost, width, width, height, cudaMemcpyHostToDevice);

    return pitch;
}

static
void _cudaMoveMemoryToHost(int *graphDevice,  int* dataHost, size_t pitch,int nvertex) {
    size_t height = nvertex;
    size_t width = height * sizeof(int);

    cudaMemcpy2D(dataHost, width, graphDevice, pitch, width, height, cudaMemcpyDeviceToHost);

    cudaFree(graphDevice);
}

void cudaBlockedFW(int *dataHost,int nvertex,int bs) {
    cudaSetDevice(0);
    //printf("%d-bb",bs);
    int *graphDevice, *predDevice;
    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, nvertex);

    int numBlock = (nvertex - 1) / bs + 1;

    dim3 gridPhase1(1 ,1, 1);
    dim3 gridPhase2(numBlock, 2 , 1);
    dim3 gridPhase3(numBlock, numBlock , 1);
    dim3 dimBlockSize(bs, bs, 1);
    unsigned shared_mem_size_dependent = (bs*bs*sizeof(int));
    unsigned shared_mem_size_partial = (2*bs*bs*sizeof(int));	

    

    for(int blockID = 0; blockID < numBlock; ++blockID) {
        _blocked_fw_dependent_ph<<<gridPhase1, dimBlockSize,shared_mem_size_dependent>>>
                (blockID, pitch / sizeof(int), nvertex, graphDevice,bs);

        _blocked_fw_partial_dependent_ph<<<gridPhase2, dimBlockSize, shared_mem_size_partial>>>
                (blockID, pitch / sizeof(int), nvertex, graphDevice,bs);

        _blocked_fw_independent_ph<<<gridPhase3, dimBlockSize,shared_mem_size_partial>>>
                (blockID, pitch / sizeof(int), nvertex, graphDevice,bs);
    }

    cudaGetLastError();
    cudaDeviceSynchronize();
    _cudaMoveMemoryToHost(graphDevice, dataHost, pitch, nvertex);
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
    cudaBlockedFW(d,n,bs);
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
