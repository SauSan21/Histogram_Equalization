#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "image.h"

#define MAX_INTENSITY 255


double get_time_diff(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

// Calculate the histogram of the image
void calculate_histogram(int histogram[], png_byte* image, int size) {
    for(int i = 0; i < size; i++) {
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
    for(int i = 0; i <= MAX_INTENSITY; i++) {
        cdf[i] = ((cdf[i] - min_cdf) * MAX_INTENSITY) / (size - min_cdf);
    }
}

// Equalize the image using the CDF
void equalize_image(png_byte* image, int cdf[], int size) {
    for(int i = 0; i < size; i++) {
        image[i] = cdf[image[i]];
    }
}

int main() {
    // Read the image file
    struct timespec start, end;
    double best_time = 0.0;
    int NUM_RUNS = 10;

    // Remove the files from previous runs
    for (int run = 0; run < NUM_RUNS; run++) {
        char filename[50];
        sprintf(filename, "equalizer%d.png", run);
        remove(filename);
    }

    if (argc < 2) {
        printf("Usage: %s <image.png>\n", argv[0]);
        return 1;
    }

    char *input_file = argv[1];
    
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
        if (run == 0 || time < best_time) {
            best_time = time;
        }

        // Create a new filename for each run
        char filename[50];
        sprintf(filename, "equalizer%d.png", run);

        // Write the equalized image to a new file each time
        write_png_file(filename, &img);

        free_image_data(&img);
    }

    printf("Best time: %f\n", best_time);


    // Free the image data
    
    return 0;
}

