#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"

#define MAX_PATH 255
#define r 5

const char *get_file_ext(char *file_path) {
    const char *p, *dot = file_path;
    while (p = strchr(dot, '.')) dot = p + 1;
    if (dot == file_path) return "";
    return dot;
}

int main(int argc, char **argv) {
    int num_threads;
    char input_file[MAX_PATH + 1], output_file[MAX_PATH + 1];
    const char *input_file_extension;
    if (argc < 2) {
        printf("Usage: ./blur input_file [output_file]\n");
        exit(1);
    }
    strncpy(input_file, argv[1], MAX_PATH);
    input_file[MAX_PATH] = '\0';
    input_file_extension = get_file_ext(input_file);

    if (argc == 2) {
        int l = strnlen(input_file, MAX_PATH) -
                strnlen(input_file_extension, MAX_PATH) - 1;
        strncpy(output_file, input_file, l);
        output_file[l] = '\0';
        strncat(strncat(output_file, "_blurred.", MAX_PATH),
                input_file_extension, MAX_PATH);
    } else {
        strncpy(output_file, argv[2], MAX_PATH);
        output_file[MAX_PATH] = '\0';
    }
    int width, height, channels;
    if (stbi_info(input_file, &width, &height, &channels) && channels != 4 &&
        channels != 3) {
        printf("Invalid input image '%s' has %d channel%s, expected 3 or 4\n",
               input_file, channels, channels > 1 ? "s" : "");
        exit(1);
    }
    uint8_t *input_img = stbi_load(input_file, &width, &height, &channels, 0);
    if (!input_img) {
        printf("Error in loading the image\n");
        exit(1);
    }
    printf(
        "Loaded image '%s' with a width of %dpx, a height of %dpx and %d "
        "channels\n",
        input_file, width, height, channels);
    size_t img_size = width * height * channels;
    uint8_t *output_img = malloc(img_size);
    if (!output_img) {
        printf("Unable to allocate memory for the output image\n");
        exit(1);
    }



 double begin = omp_get_wtime();
#pragma omp parallel num_threads(8)
    {
#pragma omp for collapse(2)
 // (largeur*hauteur)/nbr_threads iteration = le nbr de pixel traite
        for (int x_img = 0; x_img < width; x_img++) {
            for (int y_img = 0; y_img < height; y_img++) {
		// l'image sous forme d'un vecteur 
                int i_img = (y_img * width + x_img) * channels;
                int count = 0;
                int output_red = 0, output_green = 0, output_blue = 0;
		// a ne pas paralleliser 
                for (int x_box = x_img - r;
                     x_box <= x_img + r ; x_box++) {
                    for (int y_box = y_img - r;
                         y_box <= y_img + r ; y_box++) {
                        if (x_box >= 0 && x_box < width && y_box >= 0 && y_box < height) {
			   // indexer les pixels dans l' image originale (sous forme d'un vecteur)
                            int i_box = (y_box * width + x_box) * channels;
                            output_red += input_img[i_box];
                            output_green += input_img[i_box + 1];
                            output_blue += input_img[i_box + 2];
                            count++;
                        }
                    }
                }// end 3rd for
                output_img[i_img] = output_red / count;
                output_img[i_img + 1] = output_green / count;
                output_img[i_img + 2] = output_blue / count;
                if (channels == 4) output_img[i_img + 3] = input_img[i_img + 3];
            }// end 2nd for
        }// end 1st for 
#pragma omp single
       {num_threads = omp_get_num_threads(); }
    }// end parallel
    double end = omp_get_wtime();
    // milliseconds
    double time_spent = (end - begin) * 1000;




/////////////////////////////////////////////////////////////////////////////////////////////////////
    const char *output_file_extension = get_file_ext(output_file);
    if ((strcmp(output_file_extension, "jpg") ||
          strcmp(output_file_extension, "jpeg") ||
          strcmp(output_file_extension, "JPG") ||
          strcmp(output_file_extension, "JPEG")))
        stbi_write_jpg(output_file, width, height, channels, output_img, 100);
    else if (!(strcmp(output_file_extension, "bmp") ||
               strcmp(output_file_extension, "BMP")))
        stbi_write_bmp(output_file, width, height, channels, output_img);
    else
        stbi_write_png(output_file, width, height, channels, output_img,
                       width * channels);
    stbi_image_free(input_img);
    free(output_img);
    printf("Check '%s' (took %fms with %d threads)\n", output_file, time_spent,
           num_threads);
    return 0;
}