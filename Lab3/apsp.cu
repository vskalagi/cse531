#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
//#include "cuda_apsp.cuh"
#define BLOCK_SIZE 16
#define INF 200

/**
 * CUDA handle error, if error occurs print message and exit program
*
* @param error: CUDA error status
*/
#define HANDLE_ERROR(error) { \
    if (error != cudaSuccess) { \
        fprintf(stderr, "%s in %s at line %d\n", \
                cudaGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} \

/**
 * Naive CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * check if path from vertex x -> y will be short using vertex u x -> u -> y
 * for all vertices in graph
 *
 * @param u: Index of vertex u
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 * @param pred: Array of predecessors for a graph on device
 */
static __global__
void _naive_fw_kernel(const int u, size_t pitch, const int nvertex, int* const graph) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < nvertex && x < nvertex) {
        int indexYX = y * pitch + x;
        int indexUX = u * pitch + x;

        int newPath = graph[y * pitch + u] + graph[indexUX];
        int oldPath = graph[indexYX];
        if (oldPath > newPath) {
            graph[indexYX] = newPath;
        }
    }
}



/**
 * Allocate memory on device and copy memory from host to device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param graphDevice: Pointer to array of graph with distance between vertex on device
 * @param predDevice: Pointer to array of predecessors for a graph on device
 *
 * @return: Pitch for allocation
 */
static
size_t _cudaMoveMemoryToDevice(const int*  dataHost, int **graphDevice, int nvertex) {
    size_t height = nvertex;
    size_t width = height * sizeof(int);
    size_t pitch;

    // Allocate GPU buffers for matrix of shortest paths d(G) 
    HANDLE_ERROR(cudaMallocPitch(graphDevice, &pitch, width, height));

    // Copy input from host memory to GPU buffers and
    HANDLE_ERROR(cudaMemcpy2D(*graphDevice, pitch,
            dataHost, width, width, height, cudaMemcpyHostToDevice));

    return pitch;
}

/**
 * Copy memory from device to host and free device memory
 *
 * @param graphDevice: Array of graph with distance between vertex on device
 * @param predDevice: Array of predecessors for a graph on device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param pitch: Pitch for allocation
 */
static
void _cudaMoveMemoryToHost(int *graphDevice,  int* dataHost, size_t pitch,int nvertex) {
    size_t height = nvertex;
    size_t width = height * sizeof(int);

    HANDLE_ERROR(cudaMemcpy2D(dataHost, width, graphDevice, pitch, width, height, cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(graphDevice));
}

/**
 * Naive implementation of Floyd Warshall algorithm in CUDA
 *
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 */
void cudaNaiveFW(int *dataHost,int nvertex) {
    // Choose which GPU to run on, change this on a multi-GPU system.
    HANDLE_ERROR(cudaSetDevice(0));
    //int nvertex = dataHost->nvertex;

    // Initialize the grid and block dimensions here
    dim3 dimGrid((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    int *graphDevice, *predDevice;
    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, nvertex);

    cudaFuncSetCacheConfig(_naive_fw_kernel, cudaFuncCachePreferL1);
    for(int vertex = 0; vertex < nvertex; ++vertex) {
        _naive_fw_kernel<<<dimGrid, dimBlock>>>(vertex, pitch / sizeof(int), nvertex, graphDevice );
    }

    // Check for any errors launching the kernel
    HANDLE_ERROR(cudaGetLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
    _cudaMoveMemoryToHost(graphDevice, dataHost, pitch, nvertex);
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
// void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
//     HANDLE_ERROR(cudaSetDevice(0));
//     int nvertex = dataHost->nvertex;
//     int *graphDevice, *predDevice;
//     size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice, &predDevice);

//     dim3 gridPhase1(1 ,1, 1);
//     dim3 gridPhase2((nvertex - 1) / BLOCK_SIZE + 1, 2 , 1);
//     dim3 gridPhase3((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1 , 1);
//     dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);

//     int numBlock = (nvertex - 1) / BLOCK_SIZE + 1;

//     for(int blockID = 0; blockID < numBlock; ++blockID) {
//         // Start dependent phase
//         _blocked_fw_dependent_ph<<<gridPhase1, dimBlockSize>>>
//                 (blockID, pitch / sizeof(int), nvertex, graphDevice, predDevice);

//         // Start partially dependent phase
//         _blocked_fw_partial_dependent_ph<<<gridPhase2, dimBlockSize>>>
//                 (blockID, pitch / sizeof(int), nvertex, graphDevice, predDevice);

//         // Start independent phase
//         _blocked_fw_independent_ph<<<gridPhase3, dimBlockSize>>>
//                 (blockID, pitch / sizeof(int), nvertex, graphDevice, predDevice);
//     }

//     // Check for any errors launching the kernel
//     HANDLE_ERROR(cudaGetLastError());
//     HANDLE_ERROR(cudaDeviceSynchronize());
//     _cudaMoveMemoryToHost(graphDevice, predDevice, dataHost, pitch);
// }


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
    cudaNaiveFW(d,n);
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
