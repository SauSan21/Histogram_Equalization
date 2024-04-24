/**
 * Functions for reading and writing PNG images.
 * 
 * To use you need to compile with -lpng.
 */

#pragma once
#include <png.h>

typedef struct _Image {
    // always 8-bit depth (i.e. one byte per channel, values from 0 to 255)
    int width, height;
    int color_type; // one of PNG_COLOR_TYPE_GRAY, PNG_COLOR_TYPE_GRAY_ALPHA, PNG_COLOR_TYPE_RGB, PNG_COLOR_TYPE_RGB_ALPHA
    png_bytep *data;
} Image;

/**
 * Allocate memory for the image data based on the image dimensions and color type.
 * The image width, height, and color type must be set before calling this function.
 */
void malloc_image_data(Image* img);

/** Free the memory allocated for the image data. */
void free_image_data(Image* img);

/**
 * Read a PNG file into an Image struct.
 * 
 * The image data is stored in `img->data` as an array-of-array-of bytes. The data
 * must be freed with `free_image_data` when it is no longer needed.
 * 
 * The `color_type` parameter specifies the desired color type of the image data
 * and the image data is normalized/forced to this color type. It must be one of:
 * 
 *      PNG_COLOR_TYPE_GRAY       - 1 byte per pixel
 *      PNG_COLOR_TYPE_GRAY_ALPHA - 2 bytes per pixel
 *      PNG_COLOR_TYPE_RGB        - 3 bytes per pixel
 *      PNG_COLOR_TYPE_RGB_ALPHA  - 4 bytes per pixel
 * 
 * Example (reading an image as grayscale):
 *     Image img = {0};
 *     read_png_file("image.png", PNG_COLOR_TYPE_GRAY, &img);
 *     // TODO: Access the image data as img->data[y][x]...
 *     free_image_data(&img);
 */
void read_png_file(const char *filename, int color_type, Image* img);

/**
 * Write an Image struct to a PNG file.
 * 
 * Example:
 *     Image img = { .width = 256, .height = 256, .color_type = PNG_COLOR_TYPE_GRAY };
 *     malloc_image_data(&img);
 *     // TODO: Fill the image data in...
 *     write_png_file("output.png", &img);
 *     free_image_data(&img);
 */
void write_png_file(char *filename, Image* img);
