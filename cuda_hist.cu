/* 
* This program reads a PNG image and equalizes the histogram of the image using CUDA.
* The program reads the image from a file, calculates the histogram of the image, computes the CDF of 
* the histogram,normalizes the CDF and then equalizes the image using the normalized CDF. 
* The equalized image is then written to a new file.The program uses CUDA to parallelize the 
* histogram calculation, CDF computation, normalization and equalization.
* The program is timed to measure the time taken to equalize the image.
*
* The program is run multiple times and the best time is reported.
*
* The program uses the following CUDA kernels:
* 1. calculate_histogram: This kernel calculates the histogram of the image.
* 2. compute_cdf: This kernel computes the CDF of the histogram.
* 3. normalize_cdf: This kernel normalizes the CDF.
* 4. equalize: This kernel equalizes the image using the normalized CDF.
* 
* Compile the program using the following command:
* nvcc $(libpng-config --I_opts) image.c cuda_hist.cu -o cuda_hist $(libpng-config --L_opts) -lpng
*
* The program is run using the following command:
* ./cuda_hist <input_image.png> <output_image.png>
*/

#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>

#include <cuda_runtime.h>
#include "image.h"

#define MAX_INTENSITY 255

/*
* Macro to check CUDA calls
*/
#define CHECK(call)                                                       \
{                                                                         \
   const cudaError_t error = call;                                        \
   if (error != cudaSuccess)                                              \
   {                                                                      \
      printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
      printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
      exit(1);                                                            \
   }                                                                      \
}

/*
* Function to get the time difference between two timespec structures
*/
double get_time_diff(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

/*
* CUDA kernel to calculate the histogram of the image
* The kernel takes the histogram array, image array and the size of the image as input.
* The kernel calculates the histogram of the image and updates the histogram array.
* The kernel uses atomicAdd which is a thread-safe operation to update the histogram array.
*/
__global__ 
void calculate_histogram(int *histogram, png_byte *image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int i = y * width + x;
        atomicAdd(&histogram[image[i]], 1);
    }
}

/*
* CUDA kernel to compute the CDF of the histogram
* The kernel takes the histogram array, CDF array and the length of the histogram as input.
* The kernel computes the CDF of the histogram and updates the CDF array.
*/
__global__ 
void compute_cdf(int *histogram, int *cdf, int length) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += histogram[i];
        cdf[i] = sum;
    }
}

/*
* CUDA kernel to normalize the CDF
* The kernel takes the CDF array, size of the image and the minimum value of the CDF as input.
* The kernel normalizes the CDF and updates the CDF array.
*/
__global__ 
void normalize_cdf(int *cdf, int width, int height, int min_cdf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int i = y * width + x;
        if (i <= MAX_INTENSITY) {
            cdf[i] = ((cdf[i] - min_cdf) * MAX_INTENSITY) / ((width * height) - min_cdf);
        }
    }
}

/*
* CUDA kernel to equalize the image
* The kernel takes the image array, CDF array and the size of the image as input.
* The kernel equalizes the image using the normalized CDF and updates the image array.
*/
__global__ 
void equalize(png_byte *image, int *cdf, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int i = y * width + x;
        image[i] = cdf[image[i]];
    }
}

// __global__
// void mem_only(int *histogram, int *cdf, int length) {
//     extern __shared__ int temp[]; // allocated on invocation
//     int thid = threadIdx.x;

//     // load input into shared memory.
//     // This is exclusive scan, so shift right by one and set first element to 0
//     temp[thid] = (thid > 0) ? histogram[thid-1] : 0;
//     __syncthreads();

//     for (int offset = 1; offset < length; offset *= 2) {
//         if (thid >= offset) {
//             // add from a stride of 'offset' behind
//             int t = temp[thid - offset];
//             __syncthreads();
//             temp[thid] += t;
//         }
//         __syncthreads();
//     }
//     cdf[thid] = temp[thid]; // write result for this thread
// }

int main(int argc, char *argv[]) {
    // Get the start time
    struct timespec start, end;
    double total_time = 0.0;
    int NUM_RUNS = 20;

    // Check the number of arguments
    if (argc < 3) {
        printf("Usage: %s <input_image.png> <output_image.png>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];

    // Run the program multiple times and get the best time
    for (int run = 0; run < NUM_RUNS; run++) {
        // Read the image from the input file
        Image img = {0};
        read_png_file(input_file, PNG_COLOR_TYPE_GRAY, &img);

        // Calculate the histogram, CDF and equalize the image
        png_byte *image = img.data[0];
        int size = img.width * img.height;
        int width = img.width;
        int height = img.height;
        int histogram[MAX_INTENSITY + 1] = {0};
        int cdf[MAX_INTENSITY + 1] = {0};
        
        png_byte *d_image;
        int *d_cdf;
        int *d_histogram;

        clock_gettime(CLOCK_MONOTONIC, &start); // Start the timer

        // Allocate memory on the device
        CHECK(cudaMalloc(&d_histogram, (MAX_INTENSITY + 1) * sizeof(int)));
        CHECK(cudaMalloc(&d_image, size * sizeof(png_byte)));
        CHECK(cudaMalloc(&d_cdf, (MAX_INTENSITY + 1) * sizeof(int)));

        // Copy the image to the device
        CHECK(cudaMemcpy(d_histogram, histogram, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_image, image, size * sizeof(png_byte), cudaMemcpyHostToDevice));

        // Calculate the histogram of the image
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
        calculate_histogram<<<numBlocks, threadsPerBlock>>>(d_histogram, d_image, width, height);

        // Copy the histogram to the host
        CHECK(cudaMemcpy(histogram, d_histogram, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        // Calculate the CDF of the histogram
        int threadsPerBlock2 = 256;
        int blocksPerGrid = (MAX_INTENSITY + threadsPerBlock2 - 1) / threadsPerBlock2;
        compute_cdf<<<blocksPerGrid, threadsPerBlock2>>>(d_histogram, d_cdf, MAX_INTENSITY + 1);
        //mem_only<<<blocksPerGrid, threadsPerBlock>>>(d_histogram, d_cdf, MAX_INTENSITY + 1);

        // Copy the CDF to the host
        CHECK(cudaMemcpy(cdf, d_cdf, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        int min_cdf = cdf[0]; // Assuming cdf[0] is the minimum value in the CDF
        // Normalize the CDF
        normalize_cdf<<<numBlocks, threadsPerBlock>>>(d_cdf, width, height, min_cdf);

        // Equalize the image
        equalize<<<numBlocks, threadsPerBlock>>>(d_image, d_cdf, width, height);

        // Copy the equalized image to the host
        CHECK(cudaMemcpy(image, d_image, size * sizeof(png_byte), cudaMemcpyDeviceToHost));

        clock_gettime(CLOCK_MONOTONIC, &end); // get the end time
        double time_taken = get_time_diff(&start, &end); // get the time taken
        total_time += time_taken;
    
        // Free the device memory
        CHECK(cudaFree(d_histogram));
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_cdf));

        // Write the equalized image to a new file each time
        write_png_file(output_file, &img);
    }

    // Get the average time
    double avg_time = total_time / NUM_RUNS;
    printf("Average time: %f\n", avg_time);

    // Free the image data
    CHECK(cudaDeviceReset());

    return 0;

    
}
