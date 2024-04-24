/**
 * Functions for reading and writing PNG images.
 * 
 * To use you need to compile with -lpng.
 */

#include <stdlib.h>
#include <stdbool.h>

#include "image.h"

/**
 * Allocate memory for the image data based on the image dimensions and color type.
 * The image width, height, and color type must be set before calling this function.
 */
void malloc_image_data(Image* img) {
    int height = img->height;
    int width = img->width;
    int color_type = img->color_type;
    int row_bytes = width * (((color_type & PNG_COLOR_MASK_COLOR) ? 3 : 1) + ((color_type & PNG_COLOR_MASK_ALPHA) ? 1 : 0));
    img->data = (png_bytep*)malloc(sizeof(png_bytep) * height + sizeof(png_byte) * height * row_bytes);
    if (!img->data) abort();
    img->data[0] = ((png_byte*)img->data) + sizeof(png_bytep) * height;
    for (int y = 1; y < height; y++) { img->data[y] = img->data[y-1] + row_bytes; }
}

/** Free the memory allocated for the image data. */
void free_image_data(Image* img) {
    if (img) { free(img->data); }
}

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
void read_png_file(const char *filename, int color_type, Image* img) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) abort();

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);
    png_read_info(png, info);

    // Normalize the image data
    int real_color_type = png_get_color_type(png, info);
    int bit_depth = png_get_bit_depth(png, info);
    bool want_alpha = color_type & PNG_COLOR_MASK_ALPHA;
    bool want_rgb = color_type & PNG_COLOR_MASK_COLOR;
    bool has_alpha = real_color_type & PNG_COLOR_MASK_ALPHA;
    bool has_rgb = real_color_type & PNG_COLOR_MASK_COLOR;
    if (bit_depth == 16) { png_set_scale_16(png); }
    if (real_color_type == PNG_COLOR_TYPE_PALETTE) { png_set_palette_to_rgb(png); }
    if (!has_rgb && bit_depth < 8) { png_set_expand_gray_1_2_4_to_8(png); }
    if (want_alpha) {
        if (png_get_valid(png, info, PNG_INFO_tRNS)) { png_set_tRNS_to_alpha(png); }
        else if (!has_alpha) { png_set_add_alpha(png, 0xFF, PNG_FILLER_AFTER); }
    } else if (has_alpha) { png_set_strip_alpha(png); }
    if (!want_rgb && has_rgb) { png_set_rgb_to_gray_fixed(png, 1, -1, -1); }
    else if (want_rgb && !has_rgb) { png_set_gray_to_rgb(png); }

    // Check if the image is in the desired color type
    png_read_update_info(png, info);
    if (png_get_bit_depth(png, info) != 8) { fprintf(stderr, "Image bit depth is not 8\n"); abort(); }
    if (png_get_color_type(png, info) != color_type) { fprintf(stderr, "Image color type is not %d\n", color_type); abort(); }

    // Read the image data in
    img->color_type = color_type;
    img->width = png_get_image_width(png, info);
    img->height = png_get_image_height(png, info);
    malloc_image_data(img);
    png_read_image(png, img->data);

    // Cleanup
    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
}

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
void write_png_file(char *filename, Image* img) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) abort();

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) abort();

    png_infop info = png_create_info_struct(png);
    if (!info) abort();

    if (setjmp(png_jmpbuf(png))) abort();

    png_init_io(png, fp);
    png_set_IHDR(
        png, info,
        img->width, img->height,
        8, img->color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);
    png_write_image(png, img->data);
    png_write_end(png, NULL);
    fclose(fp);
    png_destroy_write_struct(&png, &info);
}

// Testing main
// int main() {
//     Image img = {0};
//     read_png_file("image.png", PNG_COLOR_TYPE_GRAY, &img);
//     write_png_file("output.png", &img);
//     free_image_data(&img);
//     return 0;
// }
