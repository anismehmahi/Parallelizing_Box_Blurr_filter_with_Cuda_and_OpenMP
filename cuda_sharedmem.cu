%%cu
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "/content/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "/content/stb_image_write.h"

#define MAX_PATH 255
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32
#define r 10

__global__ void blur_rgb(uint8_t *input_img, uint8_t *output_img, int width,
                         int height) {
    __shared__ uint8_t smem[BLOCK_WIDTH * BLOCK_HEIGHT * 3];
    int x =
        blockIdx.x * (blockDim.x - 2 * r) + threadIdx.x - r;
    int y =
        blockIdx.y * (blockDim.y - 2 * r) + threadIdx.y - r;
    x = x < 0 ? 0 : x;
    x = x >= width ? width - 1 : x;
    y = y < 0 ? 0 : y;
    y = y >= height ? height - 1 : y;
    int i_img = (y * width + x) * 3;
    int i_smem = (threadIdx.y * BLOCK_WIDTH + threadIdx.x) * 3;
    smem[i_smem] = input_img[i_img];
    smem[i_smem + 1] = input_img[i_img + 1];
    smem[i_smem + 2] = input_img[i_img + 2];
    __syncthreads();
    if (!((threadIdx.x > r - 1 &&
           threadIdx.x < BLOCK_WIDTH - r) &&
          (threadIdx.y > r - 1 &&
           threadIdx.y < BLOCK_HEIGHT - r)))
        return;
    int count = 0;
    int output_red = 0, output_green = 0, output_blue = 0;
    for (int x_box = threadIdx.x - r;
         x_box < threadIdx.x + r + 1; x_box++) {
        for (int y_box = threadIdx.y - r;
             y_box < threadIdx.y + r + 1; y_box++) {
            int i_box = (y_box * BLOCK_WIDTH + x_box) * 3;
            output_red += smem[i_box];
            output_green += smem[i_box + 1];
            output_blue += smem[i_box + 2];
            count++;
        }
    }
    output_img[i_img] = output_red / count;
    output_img[i_img + 1] = output_green / count;
    output_img[i_img + 2] = output_blue / count;
}

__global__ void blur_rgba(uint8_t *input_img, uint8_t *output_img, int width,
                          int height) {
    __shared__ uint8_t smem[BLOCK_WIDTH * BLOCK_HEIGHT * 4];
    int x =
        blockIdx.x * (blockDim.x - 2 * r) + threadIdx.x - r;
    int y =
        blockIdx.y * (blockDim.y - 2 * r) + threadIdx.y - r;
    x = x < 0 ? 0 : x;
    x = x >= width ? width - 1 : x;
    y = y < 0 ? 0 : y;
    y = y >= height ? height - 1 : y;
    int i_img = (y * width + x) * 4;
    int i_smem = (threadIdx.y * BLOCK_WIDTH + threadIdx.x) * 4;
    smem[i_smem] = input_img[i_img];
    smem[i_smem + 1] = input_img[i_img + 1];
    smem[i_smem + 2] = input_img[i_img + 2];
    smem[i_smem + 3] = input_img[i_img + 3];
    __syncthreads();
    if (!((threadIdx.x > r - 1 &&
           threadIdx.x < BLOCK_WIDTH - r) &&
          (threadIdx.y > r - 1 &&
           threadIdx.y < BLOCK_HEIGHT - r)))
        return;
    int count = 0;
    int output_red = 0, output_green = 0, output_blue = 0, output_alpha = 0;
    for (int x_box = threadIdx.x - r;
         x_box < threadIdx.x + r + 1; x_box++) {
        for (int y_box = threadIdx.y - r;
             y_box < threadIdx.y + r + 1; y_box++) {
            int i_box = (y_box * BLOCK_WIDTH + x_box) * 4;
            output_red += smem[i_box];
            output_green += smem[i_box + 1];
            output_blue += smem[i_box + 2];
            output_alpha += smem[i_box + 3];
            count++;
        }
    }
    output_img[i_img] = output_red / count;
    output_img[i_img + 1] = output_green / count;
    output_img[i_img + 2] = output_blue / count;
    output_img[i_img + 3] = output_alpha / count;
}

const char *get_file_ext(char *file_path) {
    const char *p, *dot = file_path;
    while (p = strchr(dot, '.')) dot = p + 1;
    if (dot == file_path) return "";
    return dot;
}

int main(int argc, char **argv) {
    char input_file[MAX_PATH + 1], output_file[MAX_PATH + 1];
    const char *input_file_extension;

    strncpy(input_file, "/content/2880x1800.jpg", MAX_PATH);
    input_file[MAX_PATH] = '\0';
    input_file_extension = get_file_ext(input_file);
  
        strncpy(output_file,"/content/photo1_blurr.jpg" , MAX_PATH);
        output_file[MAX_PATH] = '\0';
    
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
    uint8_t *output_img = (uint8_t *)malloc(img_size);
    if (!output_img) {
        printf("Unable to allocate memory for the output image\n");
        exit(1);
    }
    uint8_t *d_input_img, *d_output_img;
    cudaEvent_t start, stop;
    float time_spent;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaMalloc((void **)&d_input_img, img_size);
    cudaMalloc((void **)&d_output_img, img_size);
    cudaMemcpy(d_input_img, input_img, img_size, cudaMemcpyHostToDevice);
    const dim3 block_size(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
    unsigned int nb_blocksx =
        (unsigned int)(width / (BLOCK_WIDTH - 2 * r) + 1);
    unsigned int nb_blocksy =
        (unsigned int)(height / (BLOCK_HEIGHT - 2 * r) + 1);
    const dim3 grid_size(nb_blocksx, nb_blocksy, 1);
    cudaEventRecord(start, 0);
    if (channels == 3)
        blur_rgb<<<grid_size, block_size>>>(d_input_img, d_output_img, width,
                                            height);
    else
        blur_rgba<<<grid_size, block_size>>>(d_input_img, d_output_img, width,
                                             height);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Cuda error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_spent, start, stop);
    cudaMemcpy(output_img, d_output_img, img_size, cudaMemcpyDeviceToHost);
    const char *output_file_extension = get_file_ext(output_file);
    if ((strcmp(output_file_extension, "jpg") ||
          strcmp(output_file_extension, "jpeg") ||
          strcmp(output_file_extension, "JPG") ||
          strcmp(output_file_extension, "JPEG")))
        stbi_write_jpg(output_file, width, height, channels, output_img, 100);
    else if ((strcmp(output_file_extension, "bmp") ||
               strcmp(output_file_extension, "BMP")))
        stbi_write_bmp(output_file, width, height, channels, output_img);
    else
        stbi_write_png(output_file, width, height, channels, output_img,
                       width * channels);
    stbi_image_free(input_img);
    free(output_img);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input_img);
    cudaFree(d_output_img);
    printf(
        "Check '%s' (took %fms with (%d, %d) block dim and (%d, %d) grid "
        "dim)\n",
output_file, time_spent, BLOCK_WIDTH, BLOCK_HEIGHT, nb_blocksx,nb_blocksy);
    return 0;
}
