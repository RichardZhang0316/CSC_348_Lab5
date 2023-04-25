/**
 * This program implements a parallel code via a kernel call using just global memory to do a matrix multiplication 
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

__global__ void matrixMult(float *a, float *b, float *c, int width);

int main(int argc, char *argv[]) {
    int width = 0;
    srand(time(NULL)); /* seed the random number generator */
    
    /* determine if there are three arguments on the command line */
    if (argc < 3) {
        printf("Command line arguments are not enough: %s \n", argv[0]);
        return 1;
    }

    /* determine if the matrix width entered by users is legitamate */
    if (atoi(argv[2]) <= 0) {
        printf("The matrix width should not less than 1: %s \n", argv[2]);
        return 2;
    }

    /* initiate the three arrays: a and b are the input arrays, and c is the output array */
    width = atoi(argv[2]);
    float *a = (float *)malloc(width * width * sizeof(float));
    float *b = (float *)malloc(width * width * sizeof(float));
    float *c = (float *)malloc(width * width * sizeof(float));

    float *dev_a, *dev_b, *dev_c;
    int size = width * width * sizeof(float);

    /* Initialize matrices a and b */
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    /* assign random float numbers to the two input arrays */
    for (int i = 0; i < width; i++) {
      for (int j = 0; j < width; j++) {
        /* generate random numbers between 0 and 10.0 */
         a[i * width + j] = (float)rand() / RAND_MAX * 10.0;
         b[i * width + j] = (float)rand() / RAND_MAX * 10.0;
      }
    }

    /* copy the data in matrics a and b to dev_a and dev_b */
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    /* determine the size of grid and block */
    // dim3 dimGrid(1, 1);
    // dim3 dimBlock(width, width); 
    // The previous code won't work if the width exceeds 32, so we need to customize
    // the block size
    int block_size = 32;
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);


    /* to warm up the GPU */
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, width);
    cudaDeviceSynchronize(); /* make sure the first kernel call has finished before starting the timer */

    cudaEvent_t start, stop; /* measure the starting time and the ending time to calculate the time spent */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); /* start the timer */
    matrixMult<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, width);
    cudaEventRecord(stop); /* end the timer */

    cudaDeviceSynchronize();
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    /* output the execution time of the kernel function to the terminal */
    float total_time = 0.0;
    cudaEventElapsedTime(&total_time, start, stop);
    printf("Total execution time: %f seconds\n", total_time);

    /* move the output content to the output file, which is "product.dat" */
    freopen(argv[1], "w", stdout);

    /* print the output array, which is the array c */
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%f\t", c[i * width + j]);
        }
        printf("\n");
    }

    /* free memory */
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    free(a);
    free(b);
    free(c);
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
