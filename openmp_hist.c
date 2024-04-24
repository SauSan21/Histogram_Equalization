#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "image.h"

#define MAX_INTENSITY 255

// Calculate the histogram of the image
void calculate_histogram(int histogram[], png_byte* image, int size) {
    #pragma omp parallel for default(none) shared(histogram, image, size) num_threads(4)
    for(int i = 0; i < size; i++) {
        #pragma omp atomic
        histogram[image[i]]++;
    }
}

// Calculate the cumulative distribution function (CDF) of the histogram
void calculate_cdf(int cdf[], int histogram[]) {
    cdf[0] = histogram[0];
    for(int i = 1; i <= MAX_INTENSITY; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

// Normalize the CDF to the range [0, 255]
void normalize_cdf(int cdf[], int size) {
    int min_cdf = 0;
    while(cdf[min_cdf] == 0) min_cdf++;
    #pragma omp parallel for default(none) shared(cdf, size, min_cdf) num_threads(4)
    for(int i = 0; i <= MAX_INTENSITY; i++) {
        cdf[i] = ((cdf[i] - min_cdf) * MAX_INTENSITY) / (size - min_cdf);
    }
}

// Equalize the image using the CDF
void equalize_image(png_byte* image, int cdf[], int size) {
    #pragma omp parallel for default(none) shared(image, cdf, size) num_threads(4)
    for(int i = 0; i < size; i++) {
        image[i] = cdf[image[i]];
    }
}

int main() {
    // Read the image file
    Image img = {0};
    read_png_file("first.png", PNG_COLOR_TYPE_GRAY, &img);

    // Calculate the histogram, CDF, and equalize the image
    int histogram[MAX_INTENSITY + 1] = {0};
    int cdf[MAX_INTENSITY + 1] = {0};
    png_byte* image = img.data[0];
    int size = img.width * img.height;

    calculate_histogram(histogram, image, size);
    calculate_cdf(cdf, histogram);
    normalize_cdf(cdf, size);
    equalize_image(image, cdf, size);

    // Write the equalized image to a file
    write_png_file("equalized2.png", &img);

    // Free the image data
    free_image_data(&img);
    return 0;
}