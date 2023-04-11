#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define NUM_THREADS_PER_BLOCK 	256
#define NUM_BLOCKS 		16
#define PRINT_TIME 		1
#define SM_ARR_LEN		(1<<10)
#define TOL			5e-2
#define GIG                     1000000000
#define CPG                     3.07
#define IMUL(a, b) __mul24(a, b)
#define BLOCK_SIZE 16
#define TILE_WIDTH 16
#define thread_block_size 32

typedef float data_t;

void initializeArray1D(float *arr, int len, float seed);

__global__ void MMK(int width, float* Md, float* Nd, float* Pd)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    float Pvalue0 = 0;
    float Pvalue1 = 0;
    float Pvalue2 = 0;
    float Pvalue3 = 0;
    int numOfTile = SM_ARR_LEN / TILE_WIDTH;
    for (int i = 0; i < numOfTile; i++) {
        Mds[ty][tx] = Md[row * SM_ARR_LEN + (i * TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[col + (i * TILE_WIDTH + ty) * SM_ARR_LEN];
        __syncthreads();
        for(int j = 0; j < TILE_WIDTH; j = j + 4) {
            Pvalue0 += Mds[ty][j] * Nds[j][tx];
            Pvalue1 += Mds[ty][j + 1] * Nds[j + 1][tx];
            Pvalue2 += Mds[ty][j + 2] * Nds[j + 2][tx];
            Pvalue3 += Mds[ty][j + 3] * Nds[j + 3][tx];
        }
        __syncthreads();
    }
    Pd[row * SM_ARR_LEN + col] = Pvalue0 + Pvalue1 + Pvalue2 + Pvalue3;
}


void MMM_cpu_non_block(float* x, float* y, float* z) {
    for (int i = 0; i < SM_ARR_LEN; ++i){
        for (int j = 0; j < SM_ARR_LEN; ++j) {
            float sum = 0;
            for (int k = 0; k < SM_ARR_LEN; ++k) {
                float a = x[i*SM_ARR_LEN + k];
                float b = y[k*SM_ARR_LEN + j];
                sum += a * b;
            }
            z[i * SM_ARR_LEN + j] = sum;
        }
    }
}

void mmm_kij_blocked_omp(float *a, float *b, float *c) {
    data_t r;
    long int i, j, k, ii, jj, kk;
    int row_length = SM_ARR_LEN;
#pragma omp parallel shared(a0,b0,c0,row_length) private(i,j,k,r,ii,jj,kk)
    {
#pragma omp for
        for (k = 0; k < row_length; k += BLOCK_SIZE){
            for (i = 0; i < row_length; i += BLOCK_SIZE) {
                for (j = 0; j < row_length; j += BLOCK_SIZE){

                    for (kk = k; kk < k + BLOCK_SIZE; kk++) {
                        for (ii = i; ii < i + BLOCK_SIZE; ii++) {
                            r = a[ii*row_length + kk];
                            for (jj = j; jj < j + BLOCK_SIZE; jj++) {
                                c[ii*row_length + jj] +=  r* b[kk*SM_ARR_LEN + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

int compare(float* h_result, float* h_result_gold){
    int i;
    int errCount =0;
    int zeroCount = 0;
    for(i = 0; i < SM_ARR_LEN*SM_ARR_LEN; i++) {
        if (abs(h_result_gold[i] - h_result[i]) > TOL*h_result_gold[i]) {
            errCount++;
        }
        if(h_result[i]==0)
            zeroCount++;
    }
    if (zeroCount>0)
        errCount = -1;
    return errCount;
}

float errorCal(float* h_result, float* h_result_gold){
    int i;
    float error = 0;
    for(i = 0; i < SM_ARR_LEN*SM_ARR_LEN; i++) {
        if(abs(h_result_gold[i] - h_result[i])>error)
            error =  abs(h_result_gold[i] - h_result[i]);
    }
    return error;
}

double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

int main(int argc, char **argv){
    int arrLen = 0;

    // GPU Timing variables
    cudaEvent_t start, stop, start2, stop2;
    float elapsed_gpu;

    // Arrays on GPU global memoryc
    float *Md;
    float *Nd;
    float *Pd;

    // Arrays on the host memory
    float *Md_h;
    float *Pd_h;
    float *Nd_h;
    float *Pd_h_gold;
    float *Pd_h_cpu_block;
    int i, errCount = 0, zeroCount = 0;

    if (argc > 1) {
        arrLen  = atoi(argv[1]);
    }
    else {
        arrLen = SM_ARR_LEN * SM_ARR_LEN;
    }

    printf("Length of the array = %d\n", arrLen);

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // Allocate GPU memory
    size_t allocSize = arrLen * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void **)&Md, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&Pd, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&Nd, allocSize));

    // Allocate arrays on host memory
    Pd_h		           = (float *) malloc(allocSize);
    Pd_h_gold		       = (float *) malloc(allocSize);
    Md_h		           = (float *) malloc(allocSize);
    Nd_h		           = (float *) malloc(allocSize);
    Pd_h_cpu_block         = (float *) malloc(allocSize);


    // Initialize the host arrays
    printf("\nInitializing the arrays ...");
    // Arrays are initialized with a known seed for reproducability
    initializeArray1D(Md_h, arrLen, 0.53);
    initializeArray1D(Nd_h, arrLen, 0.54);
    printf("\t... done\n\n");


    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp;

    //mmm_non_block
    printf("\ncalculating results on host: ");
    clock_gettime(CLOCK_REALTIME, &time1);

    MMM_cpu_non_block(Md_h, Nd_h, Pd_h_gold);

    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp = diff(time1,time2);
    printf("%lf (msec)\n", ((double) (GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000));

    //mmm_block
    printf("\ncalculating results on cpu_MMM_block: ");
    clock_gettime(CLOCK_REALTIME, &time1);

    mmm_kij_blocked_omp(Md_h, Nd_h, Pd_h_cpu_block);

    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp = diff(time1,time2);
    printf("%lf (msec)\n", ((double) (GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000));

#if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
#endif

    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(Md, Md_h, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(Nd, Nd_h, allocSize, cudaMemcpyHostToDevice));
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2, 0);
    dim3 dimGrid(SM_ARR_LEN >> 4, SM_ARR_LEN >> 4);
    dim3 dimBlock(16,16);
    // Launch the kernel
    MMK<<<dimGrid, dimBlock>>>(SM_ARR_LEN, Md, Nd, Pd);

    // timer for kernel execution
    cudaEventRecord(stop2,0);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&elapsed_gpu, start2, stop2);
    printf("\nGPU kernel execution time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(Pd_h,Pd, allocSize, cudaMemcpyDeviceToHost));

#if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    printf("\nCompare: %d\n\n\n",compare(Pd_h,Pd_h_gold));
    printf("\nBiggest Error: %f\n\n\n",errorCal(Pd_h,Pd_h_gold));

    // Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(Pd));
    CUDA_SAFE_CALL(cudaFree(Md));
    CUDA_SAFE_CALL(cudaFree(Nd));


    free(Pd_h);
    free(Md_h);
    free(Nd_h);
    free(Pd_h_gold);

    return 0;
}

struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}


void initializeArray1D(float *arr, int len, float seed) {
    int i;
    float randNum;
    srand(seed);

    for (i = 0; i < len; i++) {
        randNum = (float) (rand()/100000);
        arr[i] = randNum;
    }
}
