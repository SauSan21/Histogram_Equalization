#include <iostream>
#include <iomanip>
#include <array>
#include <time.h>

#include <cuda_runtime.h>
#include "image.h"

#define MAX_INTENSITY 255

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

__global__ 
void calculate_histogram(int *histogram, png_byte *image, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        atomicAdd(&histogram[image[i]], 1);
    }
}

__global__ 
void compute_cdf(int *histogram, int *cdf, int length) {
    int sum = 0;
    for (int i = 0; i < length; i++) {
        sum += histogram[i];
        cdf[i] = sum;
    }
}

__global__ 
void normalize_cdf(int *cdf, int size, int min_cdf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= MAX_INTENSITY) {
        cdf[i] = ((cdf[i] - min_cdf) * MAX_INTENSITY) / (size - min_cdf);
    }
}

__global__ 
void equalize(png_byte *image, int *cdf, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        image[i] = cdf[image[i]];
    }
}

__global__
void mem_only(int *d_histogram, int *d_cdf, int size) {
    extern shared int temp[]; // allocated on invocation
    int thid = threadIdx.x;

    // load input into shared memory.
    // This is exclusive scan, so shift right by one and set first element to 0
    temp[thid] = (thid > 0) ? histogram[thid-1] : 0;
    syncthreads();

    for (int offset = 1; offset < length; offset *= 2) {
        if (thid >= offset) {
            // add from a stride of 'offset' behind
            int t = temp[thid - offset];
            syncthreads();
            temp[thid] += t;
        }
        __syncthreads();
    }
    cdf[thid] = temp[thid]; // write result for this thread
}

int main(int argc, char *argv[]) {
    struct timespec start, end;
    double best_time = 0.0;
    int NUM_RUNS = 20;

    Image img = {0};
    iif (argc < 3) {
        printf("Usage: %s <input_image.png> <output_image.png>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    
    for (int run = 0; run < NUM_RUNS; run++) {
        read_png_file(input_file, PNG_COLOR_TYPE_GRAY, &img);
        png_byte *image = img.data[0];
        int size = img.width * img.height;
        int histogram[MAX_INTENSITY + 1] = {0};
        int cdf[MAX_INTENSITY + 1] = {0};
        
        png_byte *d_image;
        int *d_cdf;
        int *d_histogram;

        clock_gettime(CLOCK_MONOTONIC, &start); // Start the timer

        CHECK(cudaMalloc(&d_histogram, (MAX_INTENSITY + 1) * sizeof(int)));
        CHECK(cudaMalloc(&d_image, size * sizeof(png_byte)));
        CHECK(cudaMalloc(&d_cdf, (MAX_INTENSITY + 1) * sizeof(int)));

        CHECK(cudaMemcpy(d_histogram, histogram, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_image, image, size * sizeof(png_byte), cudaMemcpyHostToDevice));

        calculate_histogram<<<(size + 255) / 256, 256>>>(d_histogram, d_image, size);

        CHECK(cudaMemcpy(histogram, d_histogram, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        int threadsPerBlock = 256;
        int blocksPerGrid = (MAX_INTENSITY + threadsPerBlock - 1) / threadsPerBlock;
        compute_cdf<<<blocksPerGrid, threadsPerBlock>>>(d_histogram, d_cdf, MAX_INTENSITY + 1);
        //mem_only<<<blocksPerGrid, threadsPerBlock>>>(d_histogram, d_cdf, MAX_INTENSITY + 1);

        CHECK(cudaMemcpy(cdf, d_cdf, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyDeviceToHost));

        int min_cdf = cdf[0]; // Assuming cdf[0] is the minimum value in the CDF
        normalize_cdf<<<(MAX_INTENSITY + 255) / 256, 256>>>(d_cdf, size, min_cdf);

        equalize<<<(size + 255) / 256, 256>>>(d_image, d_cdf, size);

        CHECK(cudaMemcpy(image, d_image, size * sizeof(png_byte), cudaMemcpyDeviceToHost));

        clock_gettime(CLOCK_MONOTONIC, &end); // get the end time
        double time = get_time_diff(&start, &end); // compute average difference
        if (run == 0 || time < best_time) {
            best_time = time;
        }
    

        CHECK(cudaFree(d_histogram));
        CHECK(cudaFree(d_image));
        CHECK(cudaFree(d_cdf));

        // Write the equalized image to a new file each time
        write_png_file(output_file, &img);
    }

    CHECK(cudaDeviceReset());

    return 0;

    
}
