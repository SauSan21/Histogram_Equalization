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

// __global__ 
// void exclusive_scan(int *input, png_byte *output, int length) {
//     extern __shared__ int temp[];  // allocated on invocation
//     int thid = threadIdx.x + blockIdx.x * blockDim.x;
//     int offset = 1;

//     if (thid < length) {
//         temp[thid] = input[thid]; // load input into shared memory
//     }

//     for (int d = length>>1; d > 0; d >>= 1) { // build sum in place up the tree
//         __syncthreads();
//         if (thid < d) {
//             int ai = offset*thid-1;
//             int bi = offset*(thid+1)-1;
//             temp[bi] += temp[ai];
//         }
//         offset *= 2;
//     }

//     if (thid == 0) { temp[length - 1] = 0; } // clear the last element

//     for (int d = 1; d < length; d *= 2) { // traverse down tree & build scan
//         offset >>= 1;
//         __syncthreads();
//         if (thid < d) {
//             int ai = offset*thid-1;
//             int bi = offset*(thid+1)-1;
//             int t   = temp[ai];
//             temp[ai] = temp[bi];
//             temp[bi] += t;
//         }
//     }
//     __syncthreads();

//     if (thid < length) {
//         output[thid] = temp[thid]; // write results to device memory
//     }
// }

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


int main(int argc, char *argv[]) {
    // ... (load image into "image" array and initialize "histogram" and "cdf" arrays) ...
    
    Image img = {0};
    if (argc < 2) {
        printf("Usage: %s <image.png>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    read_png_file(input_file, PNG_COLOR_TYPE_GRAY, &img);
    png_byte *image = img.data[0];
    int size = img.width * img.height;
    int histogram[MAX_INTENSITY + 1] = {0};
    int cdf[MAX_INTENSITY + 1] = {0};
    
    png_byte *d_image;
    int *d_cdf;
    int *d_histogram;
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

    CHECK(cudaMemcpy(cdf, d_cdf, (MAX_INTENSITY + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    int min_cdf = cdf[0]; // Assuming cdf[0] is the minimum value in the CDF
    normalize_cdf<<<(MAX_INTENSITY + 255) / 256, 256>>>(d_cdf, size, min_cdf);

    equalize<<<(size + 255) / 256, 256>>>(d_image, d_cdf, size);

    CHECK(cudaMemcpy(image, d_image, size * sizeof(png_byte), cudaMemcpyDeviceToHost));



    CHECK(cudaFree(d_histogram));
    CHECK(cudaFree(d_image));
    CHECK(cudaFree(d_cdf));


    // ... (save image to disk) ...

    char filename[50];
    sprintf(filename, "equalizer%d.png");

    // Write the equalized image to a new file each time
    write_png_file(filename, &img);

    return 0;

    
}
