#include <stdio.h>
#include <stdlib.h>



#define MAX_INTENSITY 255

void calculate_histogram(int histogram[], unsigned char *image, int size) {
    for(int i = 0; i < size; i++) {
        histogram[image[i]]++;
    }
}

void calculate_cdf(int cdf[], int histogram[]) {
    cdf[0] = histogram[0];
    for(int i = 1; i <= MAX_INTENSITY; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }
}

void normalize_cdf(int cdf[], int size) {
    int min_cdf = 0;
    while(cdf[min_cdf] == 0) min_cdf++;
    for(int i = 0; i <= MAX_INTENSITY; i++) {
        cdf[i] = ((cdf[i] - min_cdf) * MAX_INTENSITY) / (size - min_cdf);
    }
}

void equalize_image(unsigned char *image, int cdf[], int size) {
    for(int i = 0; i < size; i++) {
        image[i] = cdf[image[i]];
    }
}

int main() {
    // Read the image file
    Image img = {0};
 *  read_png_file("image.png", PNG_COLOR_TYPE_GRAY, &img);
    if (!img) {
        printf("Could not open the image file\n");
        return -1;
    }

    int size = img->width * img->height;
    unsigned char *image = (unsigned char *)img->imageData;
    int histogram[MAX_INTENSITY + 1] = {0};
    int cdf[MAX_INTENSITY + 1] = {0};

    calculate_histogram(histogram, image, size);
    calculate_cdf(cdf, histogram);
    normalize_cdf(cdf, size);
    equalize_image(image, cdf, size);

    // Write the equalized image to a file
    if (!cvSaveImage("output.jpg", img)) {
        printf("Could not save the image file\n");
        return -1;
    }

    cvReleaseImage(&img);
    return 0;
}













