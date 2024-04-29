/*
* This program reads an image file, calculates the histogram of the image, 
* calculates the cumulative distribution function (CDF) of the histogram, 
* normalizes the CDF to the range [0, 255], and equalizes the image using the CDF.
* The equalized image is then written to a new file.
* 
* The program is run multiple times to calculate the average time taken to equalize the image.\
* 
* Compile: gcc -Wall -O3 $(libpng-config --I_opts) image.c serial_hist.c -o serial_hist $(libpng-config --L_opts) -lpng
*
* The program takes two command-line arguments: the input image file and the output image file.
* Usage: ./serial_hist <input_image.png> <output_image.png>
* 
* The program uses the libpng library to read and write PNG files.
*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "image.h"

#define MAX_INTENSITY 255

/*
* Get the difference in time between two timespec structs in seconds.
*/
double get_time_diff(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

/*
* Calculate the histogram of the image.
* The histogram is an array where the value at index i is the number of pixels with intensity i.
* The image is a 1D array of pixels with intensity values in the range [0, 255].
* The size is the number of pixels in the image.
*/
void calculate_histogram(int histogram[], png_byte* image, int size) {
    for(int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}

/*
* Calculate the cumulative distribution function (CDF) of the histogram.
* The CDF is an array where the value at index i is the sum of the histogram values from 0 to i.
*/
void calculate_cdf(int cdf[], int histogram[]) {
    cdf[0] = histogram[0];
    for(int i = 1; i <= MAX_INTENSITY; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

/*
* Normalize the CDF to the range [0, 255].
* The CDF values are scaled to the range [0, 255] based on the minimum non-zero value in the CDF.
* This ensures that the CDF values are spread out over the entire range.
*/
void normalize_cdf(int cdf[], int size) {
    int min_cdf = 0;
    while(cdf[min_cdf] == 0) min_cdf++;
    for(int i = 0; i <= MAX_INTENSITY; i++) {
        cdf[i] = ((cdf[i] - min_cdf) * MAX_INTENSITY) / (size - min_cdf);
    }
}

/*
* Equalize the image using the CDF.
* The intensity values of the image pixels are replaced with the corresponding CDF values.
* This spreads out the intensity values over the entire range, improving the contrast of the image.
*/
void equalize_image(png_byte* image, int cdf[], int size) {
    for(int i = 0; i < size; i++) {
        image[i] = cdf[image[i]];
    }
}

int main(int argc, char *argv[]) {
    // Get the start time
    struct timespec start, end;
    double total_time = 0.0;
    int NUM_RUNS = 1000;

    // Check for the correct number of arguments
    if (argc < 3) {
        printf("Usage: %s <input_image.png> <output_image.png>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    char *output_file = argv[2];
    
    // Run the program multiple times to calculate the average time
    for (int run = 0; run < NUM_RUNS; run++) {
        // Read the image file
        Image img = {0};
        read_png_file(input_file, PNG_COLOR_TYPE_GRAY, &img);

        // Calculate the histogram, CDF, and equalize the image
        int histogram[MAX_INTENSITY + 1] = {0};
        int cdf[MAX_INTENSITY + 1] = {0};
        png_byte* image = img.data[0];
        int size = img.width * img.height;

        clock_gettime(CLOCK_MONOTONIC, &start); // get the start time

        calculate_histogram(histogram, image, size);
        calculate_cdf(cdf, histogram);
        normalize_cdf(cdf, size);
        equalize_image(image, cdf, size);

        clock_gettime(CLOCK_MONOTONIC, &end); // get the end time
        double time = get_time_diff(&start, &end); // compute average difference
        total_time += time;

        // Write the equalized image to a new file each time
        write_png_file(output_file, &img);

        // Free the image data
        free_image_data(&img);
    }

    // Calculate the average time and print it
    double avg_time = total_time / NUM_RUNS;
    printf("Average time: %f\n", avg_time);

    return 0;
}

