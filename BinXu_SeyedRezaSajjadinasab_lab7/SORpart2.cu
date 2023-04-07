#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>
#define TOL 0.00001
#define OMEGA 1.90       // TO BE DETERMINED
typedef double data_t;
typedef struct {
    long int rowlen;
    data_t *data;
} arr_rec, *arr_ptr;
#define NUM_THREADS_PER_BLOCK   256
#define NUM_BLOCKS         16
#define PRINT_TIME         1
#define SM_ARR_LEN        (1<<8)
#define TILE_WIDTH        8
#define GIG 5.0e9
#define IMUL(a, b) __mul24(a, b)
void initializeArray1D(float *arr, int len, int seed);
void SOR_blocked(arr_ptr v, int *iterations);
// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
//kernal, find the starting index of every block in the matrix
__global__ void SOR_block(float *v, int len) {
    //get the unique thread ID
    int row = threadIdx.x;
    int col = threadIdx.y;
    float result = 0;
    int threadIndex = col + row * blockDim.x;
    int increment = blockDim.x * blockDim.y;
    for (int j = 0; j < 2000; j++) {
        for (int index = threadIndex; index < len * len; index = index + increment ) {
            if (index>len && index<(len-1)*len && (index%len!=0) && (index%len!=len-1)) {
                result = v[index] - 0.25 * (v[index - 1] + v[index + 1] + v[index - len] + v[index + len]);
                //synchronizes
                __syncthreads();
                v[index] -= result * 1.9;
                __syncthreads();
            }
        }
    }
}
__global__ void SOR_strips(float *v, int len) {
    //get the unique thread ID
    int row = threadIdx.x;
    int col = threadIdx.y;
    float result = 0;
    int blockindex;
    int numOfblocks = len*len / (blockDim.x * blockDim.y);
    for(int i=0; i<2000; i++) {
        for(blockindex=0; blockindex < numOfblocks; blockindex = blockindex + 1) {
            int index = blockDim.x * blockDim.y * blockindex + row*blockDim.x + col;
            if (index>len && index<(len-1)*len && (index%len!=0) && (index%len!=len-1)) {
                result = v[index] - 0.25 * (v[index - 1] + v[index + 1] + v[index - len] + v[index + len]);
                //synchronizes
                __syncthreads();
                v[index] -= result * 1.90;
                __syncthreads();
            }
        }
    }
}

__global__ void SOR_NOT_Interleave(float *v, int len) {
    //get the unique thread ID
    int tid = threadIdx.y + threadIdx.x*blockDim.x;
    float result = 0;
    int numOfThreads   = blockDim.x*blockDim.y;
    int tileSize = len*len/numOfThreads;
    int tileDimx = len/blockDim.x;
    int tileDimy = len/blockDim.y;
    int tileCol  = tid%blockDim.x;
    int tileRow  = tid/blockDim.y;
    
    int i,j;
    for(i=0; i<1; i++) {
        int index = tileRow*tileDimy*len
                  + tileCol*tileDimx - len;
        for(j = 0; j<tileSize; j++) {
            if((j%tileDimx) == 0)
                index += len;
            else
                index += 1; 

            if (index>len && index<(len-1)*len && (index%len!=0) && (index%len!=len-1)) {
                result = v[index] - 0.25 * (v[index - 1] + v[index + 1] + v[index - len] + v[index + len]);
                //synchronizes
                __syncthreads();
                v[index] -= result * 1.90;
                __syncthreads();
            }
        }
    }
}

void SOR_cpu(float *v) {
    data_t change;
    int k, i, j;
    int length = SM_ARR_LEN;
    for(k=0;k<2000;k++){
        for (i = 1; i < length-1; i++){
            for (j = 1; j < length-1; j++) {
                change = v[i*length+j] - .25 * (v[(i-1)*length+j] +
                                                  v[(i+1)*length+j] +
                                                  v[i*length+j+1] +
                                                  v[i*length+j-1]);
                v[i*length+j] -= change * OMEGA;
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

/* -=-=-=-=- Time measurement by clock_gettime() -=-=-=-=- */
/*
  As described in the clock_gettime manpage (type "man clock_gettime" at the
  shell prompt), a "timespec" is a structure that looks like this:
 
        struct timespec {
          time_t   tv_sec;   // seconds
          long     tv_nsec;  // and nanoseconds
        };
 */

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
/*
     This method does not require adjusting a #define constant

  How to use this method:

      struct timespec time_start, time_stop;
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
      // DO SOMETHING THAT TAKES TIME
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
      measurement = interval(time_start, time_stop);

 */


/* This routine "wastes" a little time to make sure the machine gets
   out of power-saving mode (800 MHz) and switches to normal speed. */
double wakeup_delay()
{
  double meas = 0; int i, j;
  struct timespec time_start, time_stop;
  double quasi_random = 0;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
  j = 100;
  while (meas < 1.0) {
    for (i=1; i<j; i++) {
      /* This iterative calculation uses a chaotic map function, specifically
         the complex quadratic map (as in Julia and Mandelbrot sets), which is
         unpredictable enough to prevent compiler optimisation. */
      quasi_random = quasi_random*quasi_random - 1.923432;
    }
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
    meas = interval(time_start, time_stop);
    j *= 2; /* Twice as much delay next time, until we've taken 1 second */
  }
  return quasi_random;
}
int main(int argc, char **argv) {
        int arrLen = 0;
    float final_answer;
    final_answer = wakeup_delay();
    //GPU timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;

    //arrays in GPU global memoryc
    float *d_x;

    //arrays on host memory
    float *h_x;
    float *h_y_cpu;
    float *h_y_gpu;

//    int i, errCount = 0, zeroCount = 0;

    if (argc > 1) {
        arrLen = atoi(argv[1]);
    } else {
        arrLen = SM_ARR_LEN * SM_ARR_LEN;
    }


    

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // Allocate GPU memory, d_x-destination
    size_t allocSize = arrLen * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, allocSize));
    // Allocate arrays on host memory
    h_x                        = (float *) malloc(allocSize);
    h_y_cpu                    = (float *) malloc(allocSize);
    h_y_gpu                    = (float *) malloc(allocSize);
    // Initialize the host arrays, h_x-source
    printf("\nInitializing the arrays ...");
    // Arrays are initialized with a known seed for reproducability
    initializeArray1D(h_x, arrLen, 2000);
    initializeArray1D(h_y_cpu, arrLen, 2000);
    initializeArray1D(h_y_gpu, arrLen, 2000);
    printf("\t... done\n\n");

    int i;
    for(i=0; i<SM_ARR_LEN*SM_ARR_LEN; i++){
            h_y_cpu[i] = h_x[i];
    }

    struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp;
#define GIG 5.0e9
    // Compute the results on the host
    printf("\ncalculating results on host: ");  

    clock_gettime(CLOCK_REALTIME, &time1);
    
    SOR_cpu(h_y_cpu);

    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp = diff(time1,time2);
    printf("%lf (msec)\n", ((double) (GIG * time_stamp.tv_sec + time_stamp.tv_nsec)/1000000));

//for blocks
#if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
#endif
    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
//    CUDA_SAFE_CALL(cudaMemcpy(d_y, h_y, allocSize, cudaMemcpyHostToDevice));
    //launch the kernel(function)
    dim3 dimBlock(16, 16);
    SOR_block<<<1, dimBlock>>>(d_x, SM_ARR_LEN);
    //check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    //transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_y_gpu, d_x, allocSize, cudaMemcpyDeviceToHost));
#if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time block: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

printf("\nCompare: %d\n\n\n",compare(h_y_gpu,h_y_cpu));

    // for block strips
#if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
#endif
    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
//    dim3 dimGrid();
    dim3 dimBlockStrip(1, 16 * 16);
    //launch the kernel(function)
    SOR_strips<<<1, dimBlockStrip>>>(d_x, SM_ARR_LEN);
    //check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    //transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_y_gpu, d_x, allocSize, cudaMemcpyDeviceToHost));
#if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time strip: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

printf("\nCompare: %d\n\n\n",compare(h_y_gpu,h_y_cpu));

/*
    // for interleave
#if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
#endif
    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(d_x, h_x, allocSize, cudaMemcpyHostToDevice));
//    dim3 dimGrid();
    dim3 blockDim(16, 16, 1);
    //launch the kernel(function)
    SOR_NOT_Interleave<<<1, blockDim>>>(d_x, SM_ARR_LEN);
    //check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());
    //transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(h_y_gpu, d_x, allocSize, cudaMemcpyDeviceToHost));
#if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nGPU time: %f (msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

printf("\nCompare: %d\n\n\n",compare(h_y_gpu,h_y_cpu));
*/
    //free device and host memory
    CUDA_SAFE_CALL(cudaFree(d_x));
    free(h_x);
    free(h_y_gpu);
    free(h_y_cpu);

    printf("\n");
  printf("Initial delay was calculating: %g \n", final_answer);
    return 0;
}
void initializeArray1D(float *arr, int len, int seed) {
    int i;
    float randNum;
    srand(seed);
    for (i = 0; i < len; i++) {
        randNum = (float) rand();
        arr[i] = randNum;
    }
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