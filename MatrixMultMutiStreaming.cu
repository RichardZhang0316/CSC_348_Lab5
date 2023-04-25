/**
 * This program implements a parallel code via a kernel call using muti-streaming to do a matrix multiplication 
 * and prints out the execution time and the product of the two matrices with each element printed out to a 
 * file called “product.dat” in a tab-delimited, row/column format.
 * 
 * This is the original matrixMult program, and it would be compared with other programs using various optimization methods
 * later.
 *
 * Users are expected to enter three arguments: the executable file, the output file (which is product.dat), and
 * the width of the square matrics.
 *
 * @author Richard Zhang {zhank20@wfu.edu}
 * @date Apr.25, 2023
 * @assignment Lab 5
 * @course CSC 347
 **/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_STREAMS 4

__global__ void matrixMult(float *a, float *b, float *c, int width);

int main(int argc, char *argv[]) {
    int width = 0;
    srand(time(NULL)); // seed the random number generator

    // Determine if there are three arguments on the command line
    if (argc < 3) {
        printf("Command line arguments are not enough: %s \n", argv[0]);
        return 1;
    }

    // Determine if the matrix width entered by users is legitimate
    if (atoi(argv[2]) <= 0) {
        printf("The matrix width should not be less than 1: %s \n", argv[2]);
        return 2;
    }

    // Initialize the three arrays: a and b are the input arrays, and c is the output array
    width = atoi(argv[2]);
    float *a = (float *)malloc(width * width * sizeof(float));
    float *b = (float *)malloc(width * width * sizeof(float));
    float *c = (float *)malloc(width * width * sizeof(float));

    float *dev_a, *dev_b, *dev_c;
    int size = width * width * sizeof(float);

    // Initialize matrices a and b
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Assign random float numbers to the two input arrays
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            // generate random numbers between 0 and 10.0
            a[i * width + j] = (float)rand() / RAND_MAX * 10.0;
            b[i * width + j] = (float)rand() / RAND_MAX * 10.0;
        }
    }

    // Create an array of CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate the size of submatrices
    int sub_width = width / NUM_STREAMS;
    int sub_size = sub_width * width * sizeof(float);

    /* determine the size of warm up grid and block */
    int block_size_warmup = 32;
    dim3 dimBlock_warmup(block_size_warmup, block_size_warmup);
    dim3 dimGrid_warmup((width + dimBlock_warmup.x - 1) / dimBlock_warmup.x, (width + dimBlock_warmup.y - 1) / dimBlock_warmup.y);
    /* to warm up the GPU */
    matrixMult<<<dimGrid_warmup, dimBlock_warmup>>>(dev_a, dev_b, dev_c, width);
    cudaDeviceSynchronize(); /* make sure the first kernel call has finished before starting the timer */

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); /* start the timer */

    // Copy input matrices to GPU and perform matrix multiplication using multiple streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMemcpyAsync(&dev_a[i * sub_width * width], &a[i * sub_width * width], sub_size, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&dev_b[i * sub_width], &b[i * sub_width], sub_size, cudaMemcpyHostToDevice, streams[i]);

        int block_size = 32;
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

        matrixMult<<<dimGrid, dimBlock, 0, streams[i]>>>(dev_a, dev_b, dev_c, width);
    }

    cudaEventRecord(stop); /* end the timer */

    // Copy the result back to the host using multiple streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMemcpyAsync(&c[i * sub_width * width], &dev_c[i * sub_width * width], sub_size, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    /* output the execution time of the kernel function to the terminal */
    float total_time = 0.0;
    cudaEventElapsedTime(&total_time, start, stop);
    printf("Total execution time: %f seconds\n", total_time);
    
    // Move the output content to the output file, which is "product.dat"
    freopen(argv[1], "w", stdout);

    // Print the output array, which is the array c
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f\t", c[i * width + j]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(a);
    free(b);
    free(c);

    // Destroy CUDA streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}

__global__ void matrixMult(float *a, float *b, float *c, int width) {
    float sum = 0;
    int k = 0;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (col < width && row < width) {
        for (k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }

}
