//
// Created by hyunho on 9/2/19.
//
#include <time.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
#include "./vpx_dsp_rtcd.h"
#include <vpx_dsp/psnr.h>
#include <vpx_dsp/vpx_dsp_common.h>
#include <vpx_scale/yv12config.h>
#include <vpx_mem/vpx_mem.h>
#include <sys/param.h>
#include <math.h>
#include <libgen.h>

#include "third_party/libyuv/include/libyuv/convert.h"
#include "third_party/libyuv/include/libyuv/convert_from.h"
//#include "third_party/libyuv/include/libyuv/scale.h"

#include "vpx/vpx_palantir.h"

#if CONFIG_SNPE

#include <vpx/snpe/main.hpp>

#endif

#ifdef __ANDROID_API__

#include <android/log.h>
#include <arm_neon.h>

#define TAG "LoadInputTensor JNI"
#define _UNKNOWN   0
#define _DEFAULT   1
#define _VERBOSE   2
#define _DEBUG    3
#define _INFO        4
#define _WARN        5
#define _ERROR    6
#define _FATAL    7
#define _SILENT       8
#define LOGUNK(...) __android_log_print(_UNKNOWN,TAG,__VA_ARGS__)
#define LOGDEF(...) __android_log_print(_DEFAULT,TAG,__VA_ARGS__)
#define LOGV(...) __android_log_print(_VERBOSE,TAG,__VA_ARGS__)
#define LOGD(...) __android_log_print(_DEBUG,TAG,__VA_ARGS__)
#define LOGI(...) __android_log_print(_INFO,TAG,__VA_ARGS__)
#define LOGW(...) __android_log_print(_WARN,TAG,__VA_ARGS__)
#define LOGE(...) __android_log_print(_ERROR,TAG,__VA_ARGS__)
#define LOGF(...) __android_log_print(_FATAL,TAG,__VA_ARGS__)
#define LOGS(...) __android_log_print(_SILENT,TAG,__VA_ARGS__)
#endif


palantir_cfg_t *init_palantir_cfg() {
    palantir_cfg_t *config = (palantir_cfg_t *) vpx_calloc(1, sizeof(palantir_cfg_t));
    return config;
}

void remove_palantir_cfg(palantir_cfg_t *config) {
    if (config) {
        remove_palantir_dnn(config->dnn);
        remove_palantir_cache_profile(config->cache_profile);
        remove_bilinear_coeff(config->bilinear_coeff);
        vpx_free(config);
    }
}

palantir_dnn_t *init_palantir_dnn(int scale) {
    palantir_dnn_t *profile = (palantir_dnn_t *) vpx_calloc(1, sizeof(palantir_dnn_t));
    profile->interpreter = NULL;
    profile->scale = scale;

    return profile;
}

void remove_palantir_dnn(palantir_dnn_t *dnn) {
    if (dnn) {
        if (dnn->interpreter) {
#if CONFIG_SNPE
            snpe_free(dnn->interpreter);
#endif
        }
        vpx_free(dnn);
    }
}

palantir_cache_profile_t *init_palantir_cache_profile() {
    palantir_cache_profile_t *profile = (palantir_cache_profile_t *) vpx_calloc(1, sizeof(palantir_cache_profile_t));
    profile->file = NULL;
    profile->num_dummy_bits = 0;

    return profile;
}

void remove_palantir_cache_profile(palantir_cache_profile_t *cache_profile) {
    if (cache_profile) {
        if (cache_profile->file) fclose(cache_profile->file);
        if (cache_profile->block_apply_dnn) vpx_free(cache_profile->block_apply_dnn);
        if (cache_profile->anchor_block_input_buffer) {
            RGB24_free_frame_buffer(cache_profile->anchor_block_input_buffer);
            vpx_free(cache_profile->anchor_block_input_buffer);
        }
        if (cache_profile->anchor_block_sr_buffer) {
            RGB24_free_frame_buffer(cache_profile->anchor_block_sr_buffer);
            vpx_free(cache_profile->anchor_block_sr_buffer);
        }
        vpx_free(cache_profile);
    }
}

int read_cache_profile_dummy_bits(palantir_cache_profile_t *cache_profile) {
    int i, dummy;

    if (cache_profile == NULL) {
        fprintf(stderr, "%s: cache_profile is NULL", __func__);
        return -1;
    }
    if (cache_profile->file == NULL) {
        fprintf(stderr, "%s: cache_profile->file is NULL", __func__);
        return -1;
    }

    fprintf(stdout, "cache_profile->num_dummy_bits=%d", cache_profile->num_dummy_bits);
    if (cache_profile->num_dummy_bits > 0) {
        for (i = 0; i < cache_profile->num_dummy_bits; i++) {
            cache_profile->offset_byte = (cache_profile->offset_byte+1)%8;
            cache_profile->offset ++;
        }
    }

    if (fread(&cache_profile->num_dummy_bits, sizeof(int), 1, cache_profile->file) != 1) {
        fprintf(stderr, "%s: fail to read a cache profile\n", __func__);
        return -1;
    } else {
        cache_profile->offset += 32;
    }
    fprintf(stdout, "cache_profile->num_dummy_bits=%d", cache_profile->num_dummy_bits);

    return 0;
}

int read_cache_profile(palantir_cache_profile_t *profile, int num_patches_per_row, int num_patches_per_column, int save_finegrained_metadata, int save_super_finegrained_metadata) {

    fprintf(stdout, "Executing read_cache_profile");

    int frame_apply_dnn = 0;
    if (profile == NULL) {
        fprintf(stderr, "%s: profile is NULL", __func__);
        return -1;
    }
    if (profile->file == NULL) {
        fprintf(stderr, "%s: profile->file is NULL", __func__);
        return -1;
    }

    const int remaining_bits = num_patches_per_row * num_patches_per_column;
    for (int idx = 0; idx < remaining_bits; idx ++) {
        fprintf(stdout, "idx=%d", idx);
        if (profile->offset_byte == 0){
            if (fread(&profile->byte_value, sizeof(uint8_t), 1, profile->file) != 1) {
                    fprintf(stderr, "%s: fail to read a cache profile\n", __func__);
                    return -1;
            }
        }
        fprintf(stdout, "idx=%d", idx);
        profile->block_apply_dnn[idx] = (profile->byte_value & (1 << (profile->offset % 8))) >> (profile->offset % 8);
        fprintf(stdout, "idx=%d", idx);
        if (profile->block_apply_dnn[idx]!=0) {
            frame_apply_dnn = 1;
        }
        fprintf(stdout, "idx=%d", idx);
        profile->offset_byte = (profile->offset_byte+1)%8;
        profile->offset ++;
        fprintf(stdout, "idx=%d", idx);
    }

    if (save_finegrained_metadata || save_super_finegrained_metadata) {
        for (int idx = 0; idx < remaining_bits; idx ++) {
            profile->block_apply_dnn[idx] = 0;
        }
        frame_apply_dnn = 0;
    }

    return frame_apply_dnn;
}

void remove_palantir_worker(palantir_worker_data_t *mwd, int num_threads) {
    int i;
    if (mwd != NULL) {
        for (i = 0; i < num_threads; ++i) {
            vpx_free_frame_buffer(mwd[i].lr_resiudal);
            vpx_free(mwd[i].lr_resiudal);

            //free decode block lists
            palantir_interp_block_t *intra_block = mwd[i].intra_block_list->head;
            palantir_interp_block_t *prev_block = NULL;
            while (intra_block != NULL) {
                prev_block = intra_block;
                intra_block = intra_block->next;
                vpx_free(prev_block);
            }
            vpx_free(mwd[i].intra_block_list);

            palantir_interp_block_t *inter_block = mwd[i].inter_block_list->head;
            while (inter_block != NULL) {
                prev_block = inter_block;
                inter_block = inter_block->next;
                vpx_free(prev_block);
            }
            vpx_free(mwd[i].inter_block_list);

            if (mwd[i].latency_log != NULL) fclose(mwd[i].latency_log);
            if (mwd[i].metadata_log != NULL) fclose(mwd[i].metadata_log);
        }
        vpx_free(mwd);
    }
}

void update_palantir_dependency_graph_block_residual_uint8(palantir_cfg_t *palantir_cfg, uint8_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling) {
    palantir_dependency_graph_t *palantir_dependency_graph = palantir_cfg->dependency_graph;

    float *num_pixels_per_patch = vpx_calloc(palantir_cfg->num_patches_per_row*palantir_cfg->num_patches_per_column, sizeof(float));
    float *mean_per_patch = vpx_calloc(palantir_cfg->num_patches_per_row*palantir_cfg->num_patches_per_column, sizeof(float));
    float *std_per_patch = vpx_calloc(palantir_cfg->num_patches_per_row*palantir_cfg->num_patches_per_column, sizeof(float));

    int max_y_offset = ((y_offset + height)<<y_subsampling) <= palantir_cfg->num_patches_per_column*palantir_cfg->patch_height? y_offset + height : ((palantir_cfg->num_patches_per_column*palantir_cfg->patch_height) >> y_subsampling);
    int max_x_offset = ((x_offset + width)<<x_subsampling) <= palantir_cfg->num_patches_per_row*palantir_cfg->patch_width? x_offset + width : ((palantir_cfg->num_patches_per_row*palantir_cfg->patch_width) >> x_subsampling);
    uint8_t *originalBuffer = buffer;
    if (y_offset < 0) {
        originalBuffer += (stride*(-y_offset));
    }
    if (x_offset < 0) {
        originalBuffer += (-x_offset);
    }
    y_offset = y_offset >= 0? y_offset:0;
    x_offset = x_offset >= 0? x_offset:0;


    buffer = originalBuffer;
    for (int y =  y_offset; y < max_y_offset; y ++) {
        int row_idx = (y<<y_subsampling) / palantir_cfg->patch_height;
        for (int x = x_offset; x < max_x_offset; x ++) {
            int col_idx = (x<<x_subsampling) / palantir_cfg->patch_width;
            num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx] += 1;
            mean_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx] += buffer[x];
        }
        buffer += stride;
    }

    for (int row_idx=0; row_idx < palantir_cfg->num_patches_per_column; row_idx ++) {
        for (int col_idx = 0; col_idx < palantir_cfg->num_patches_per_row; col_idx ++) {
            if (num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx]==0) {
                continue;
            }
            mean_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx] /= num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx];
        }
    }

    buffer = originalBuffer;
    for (int y = y_offset; y < max_y_offset; y ++) {
        int row_idx = (y<<y_subsampling) / palantir_cfg->patch_height;
        for (int x = x_offset; x < max_x_offset; x ++) {
            int col_idx = (x<<x_subsampling) / palantir_cfg->patch_width;
            std_per_patch[row_idx * palantir_cfg->num_patches_per_row + col_idx] += powf(buffer[x]-mean_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx], 2);
        }
        buffer += stride;
    }

    for (int row_idx=0; row_idx < palantir_cfg->num_patches_per_column; row_idx ++) {
        for (int col_idx = 0; col_idx < palantir_cfg->num_patches_per_row; col_idx ++) {
            if (num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx]==0) {
                continue;
            }
            palantir_dependency_graph->block_residual[row_idx*palantir_cfg->num_patches_per_row+col_idx] += std_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx];
        }
    }

    vpx_free(num_pixels_per_patch);
    vpx_free(mean_per_patch);
    vpx_free(std_per_patch);
}

void update_palantir_dependency_graph_block_residual_int16(palantir_cfg_t *palantir_cfg, int16_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling) {
    palantir_dependency_graph_t *palantir_dependency_graph = palantir_cfg->dependency_graph;

    float *num_pixels_per_patch = vpx_calloc(palantir_cfg->num_patches_per_row*palantir_cfg->num_patches_per_column, sizeof(float));
    float *mean_per_patch = vpx_calloc(palantir_cfg->num_patches_per_row*palantir_cfg->num_patches_per_column, sizeof(float));
    float *std_per_patch = vpx_calloc(palantir_cfg->num_patches_per_row*palantir_cfg->num_patches_per_column, sizeof(float));

    int max_y_offset = ((y_offset + height)<<y_subsampling) <= palantir_cfg->num_patches_per_column*palantir_cfg->patch_height? y_offset + height : ((palantir_cfg->num_patches_per_column*palantir_cfg->patch_height) >> y_subsampling);
    int max_x_offset = ((x_offset + width)<<x_subsampling) <= palantir_cfg->num_patches_per_row*palantir_cfg->patch_width? x_offset + width : ((palantir_cfg->num_patches_per_row*palantir_cfg->patch_width) >> x_subsampling);
    uint8_t *originalBuffer = buffer;
    if (y_offset < 0) {
        originalBuffer += (stride*(-y_offset));
    }
    if (x_offset < 0) {
        originalBuffer += (-x_offset);
    }
    y_offset = y_offset >= 0? y_offset:0;
    x_offset = x_offset >= 0? x_offset:0;


    buffer = originalBuffer;
    for (int y =  y_offset; y < max_y_offset; y ++) {
        int row_idx = (y<<y_subsampling) / palantir_cfg->patch_height;
        for (int x = x_offset; x < max_x_offset; x ++) {
            int col_idx = (x<<x_subsampling) / palantir_cfg->patch_width;
            num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx] += 1;
            mean_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx] += buffer[x];
        }
        buffer += stride;
    }

    for (int row_idx=0; row_idx < palantir_cfg->num_patches_per_column; row_idx ++) {
        for (int col_idx = 0; col_idx < palantir_cfg->num_patches_per_row; col_idx ++) {
            if (num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx]==0) {
                continue;
            }
            mean_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx] /= num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx];
        }
    }

    buffer = originalBuffer;
    for (int y = y_offset; y < max_y_offset; y ++) {
        int row_idx = (y<<y_subsampling) / palantir_cfg->patch_height;
        for (int x = x_offset; x < max_x_offset; x ++) {
            int col_idx = (x<<x_subsampling) / palantir_cfg->patch_width;
            std_per_patch[row_idx * palantir_cfg->num_patches_per_row + col_idx] += powf(buffer[x]-mean_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx], 2);
        }
        buffer += stride;
    }

    for (int row_idx=0; row_idx < palantir_cfg->num_patches_per_column; row_idx ++) {
        for (int col_idx = 0; col_idx < palantir_cfg->num_patches_per_row; col_idx ++) {
            if (num_pixels_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx]==0) {
                continue;
            }
            palantir_dependency_graph->block_residual[row_idx*palantir_cfg->num_patches_per_row+col_idx] += std_per_patch[row_idx*palantir_cfg->num_patches_per_row+col_idx];
        }
    }

    vpx_free(num_pixels_per_patch);
    vpx_free(mean_per_patch);
    vpx_free(std_per_patch);
}

// Bicubic interpolation kernel
static float u(float s, float a) {
    if (fabs(s) >= 0 && fabs(s) <= 1) {
        return (a + 2) * pow(fabs(s), 3) - (a + 3) * pow(fabs(s), 2) + 1;
    } else if (fabs(s) > 1 && fabs(s) <= 2) {
        return a * pow(fabs(s), 3) - 5 * a * pow(fabs(s), 2) + 8 * a * fabs(s) - 4 * a;
    }
    return 0;
}

// Function to add edge padding to an image
static void addEdgePadding(int16_t* input, int input_stride, int16_t* output, int width, int height) {
    const int paddingSize = 2;
    int paddedWidth = width + 2 * paddingSize;
    int paddedHeight = height + 2 * paddingSize;

    // Initialize the output image with zeros
    memset(output, 0, sizeof(int16_t) * paddedWidth * paddedHeight);

    // Copy the input image into the center of the output image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            output[(y + paddingSize) * paddedWidth + (x + paddingSize)] = input[y * input_stride + x];
        }
    }

    // Fill the edges
    // Top and bottom
    for (int x = 0; x < width; x++) {
        for (int p = 0; p < paddingSize; p++) {
            // Top edge
            output[p * paddedWidth + (x + paddingSize)] = input[x];
            // Bottom edge
            output[(height + paddingSize + p) * paddedWidth + (x + paddingSize)] = input[(height - 1) * input_stride + x];
        }
    }
    // Left and right
    for (int y = 0; y < height; y++) {
        for (int p = 0; p < paddingSize; p++) {
            // Left edge
            output[(y + paddingSize) * paddedWidth + p] = input[y * input_stride];
            // Right edge
            output[(y + paddingSize) * paddedWidth + (width + paddingSize + p)] = input[y * input_stride + (width - 1)];
        }
    }
    // Corners
    for (int p = 0; p < paddingSize; p++) {
        for (int q = 0; q < paddingSize; q++) {
            // Top-left corner
            output[p * paddedWidth + q] = input[0];
            // Top-right corner
            output[p * paddedWidth + (width + paddingSize + q)] = input[width - 1];
            // Bottom-left corner
            output[(height + paddingSize + p) * paddedWidth + q] = input[(height - 1) * input_stride];
            // Bottom-right corner
            output[(height + paddingSize + p) * paddedWidth + (width + paddingSize + q)] = input[(height - 1) * input_stride + (width - 1)];
        }
    }
}

// Bicubic resize function
static void bicubicResize(int16_t* input, int16_t* output, int width, int height, float a) {
    int dH = height>>1;
    int dW = width>>1;
    memset(output, 0, sizeof(int16_t) * dH * dW);

    int h = 2;
    float matl[4] = {u(1,a),u(0,a),u(1,a),u(2,a)};
    float matr[4] = {u(1,a),u(0,a),u(1,a),u(2,a)};

    for (int j = 0; j < dH; j ++) {
        for (int i = 0; i < dW; i ++) {
            int x = i * 2 + 2, y = j * 2 + 2;
            int x1 = 1, x2 = 0, x3 = 1, x4 = 2;
            int y1 = 1, y2 = 0, y3 = 1, y4 = 2;

            float matm[4][4] = {
                {input[(y-y1)*(width+4)+(x-x1)], input[(y-y2)*(width+4)+(x-x1)], input[(y+y3)*(width+4)+(x-x1)], input[(y+y4)*(width+4)+(x-x1)]},
                {input[(y-y1)*(width+4)+(x-x2)], input[(y-y2)*(width+4)+(x-x2)], input[(y+y3)*(width+4)+(x-x2)], input[(y+y4)*(width+4)+(x-x2)]},
                {input[(y-y1)*(width+4)+(x+x3)], input[(y-y2)*(width+4)+(x+x3)], input[(y+y3)*(width+4)+(x+x3)], input[(y+y4)*(width+4)+(x+x3)]},
                {input[(y-y1)*(width+4)+(x+x4)], input[(y-y2)*(width+4)+(x+x4)], input[(y+y3)*(width+4)+(x+x4)], input[(y+y4)*(width+4)+(x+x4)]}
            };
            float result = 0.0;
            for (int i = 0; i < 4; i++) {
                float temp = 0.0;
                for (int j = 0; j < 4; j++) {
                    temp += matl[j] * matm[j][i];
                }
                result += temp * matr[i];
            }
            output[j*dW+i] = result;
        }
    }
}

void update_palantir_dependency_graph_block_residual_v2_int16(palantir_cfg_t *palantir_cfg, int16_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling) {
    palantir_dependency_graph_t *palantir_dependency_graph = palantir_cfg->dependency_graph;

    if (x_offset < 0 || y_offset < 0) {
        return;
    }

    int max_y_offset = ((y_offset + height)<<y_subsampling) <= palantir_cfg->num_patches_per_column*palantir_cfg->patch_height? y_offset + height : ((palantir_cfg->num_patches_per_column*palantir_cfg->patch_height) >> y_subsampling);
    int max_x_offset = ((x_offset + width)<<x_subsampling) <= palantir_cfg->num_patches_per_row*palantir_cfg->patch_width? x_offset + width : ((palantir_cfg->num_patches_per_row*palantir_cfg->patch_width) >> x_subsampling);
    width = max_x_offset - x_offset;
    if (width <= 0) {
        return;
    }
    width = (width>>1)<<1;
    max_x_offset = x_offset + width;
    height = max_y_offset - y_offset;
    if (height <= 0) {
        return;
    }
    height = (height>>1)<<1;
    max_y_offset = y_offset + height;
    
    int16_t *padded_input = vpx_calloc((width+2*2)*(height+2*2), sizeof(int16_t));
    addEdgePadding(buffer, stride, padded_input, width, height);
    
    int16_t *downscaled = vpx_calloc((width>>1)*(height>>1), sizeof(int16_t));
    bicubicResize(padded_input, downscaled, width, height, -0.5);

    int16_t *reconstructed = vpx_calloc(width*height, sizeof(int16_t));
    vpx_bilinear_interp_int16_int16_c(downscaled, width>>1, reconstructed, width, 0, 0, width>>1, height>>1, 2, palantir_cfg->bilinear_coeff);

    for (int i = 0; i < height; i ++) {
        int row_idx = ((y_offset+i)<<y_subsampling) / palantir_cfg->patch_height;
        for (int j = 0; j < width; j ++) {
            int col_idx = ((x_offset+j)<<x_subsampling) / palantir_cfg->patch_width;
            palantir_cfg->dependency_graph->block_residual[row_idx*palantir_cfg->num_patches_per_row+col_idx] += powf(buffer[(i+y_offset)*stride+j+x_offset]-reconstructed[i*width+j], 2);
        }
    }

    vpx_free(padded_input);
    vpx_free(downscaled);
    vpx_free(reconstructed);
}

void update_palantir_dependency_graph_block_residual_v2_uint8(palantir_cfg_t *palantir_cfg, uint8_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling) {
    palantir_dependency_graph_t *palantir_dependency_graph = palantir_cfg->dependency_graph;

    if (x_offset < 0 || y_offset < 0) {
        return;
    }

    int max_y_offset = ((y_offset + height)<<y_subsampling) <= palantir_cfg->num_patches_per_column*palantir_cfg->patch_height? y_offset + height : ((palantir_cfg->num_patches_per_column*palantir_cfg->patch_height) >> y_subsampling);
    int max_x_offset = ((x_offset + width)<<x_subsampling) <= palantir_cfg->num_patches_per_row*palantir_cfg->patch_width? x_offset + width : ((palantir_cfg->num_patches_per_row*palantir_cfg->patch_width) >> x_subsampling);
    width = max_x_offset - x_offset;
    if (width <= 0) {
        return;
    }
    width = (width>>1)<<1;
    max_x_offset = x_offset + width;
    height = max_y_offset - y_offset;
    if (height <= 0) {
        return;
    }
    height = (height>>1)<<1;
    max_y_offset = y_offset + height;

    int16_t *input_int16 = vpx_calloc(width*height, sizeof(int16_t));
    for (int i = 0; i < height; i ++) {
        for (int j = 0; j < width; j ++) {
            input_int16[i*width+j] = buffer[(i+y_offset)*stride+j+x_offset];
        }
    }

    int16_t *padded_input = vpx_calloc((width+2*2)*(height+2*2), sizeof(int16_t));
    addEdgePadding(input_int16, width, padded_input, width, height);
    
    int16_t *downscaled = vpx_calloc((width>>1)*(height>>1), sizeof(int16_t));
    bicubicResize(padded_input, downscaled, width, height, -0.5);

    uint8_t *reconstructed = vpx_calloc(width*height, sizeof(uint8_t));
    vpx_bilinear_interp_int16_c(downscaled, width>>1, reconstructed, width, 0, 0, width>>1, height>>1, 2, palantir_cfg->bilinear_coeff);

    for (int i = 0; i < height; i ++) {
        int row_idx = ((y_offset+i)<<y_subsampling) / palantir_cfg->patch_height;
        for (int j = 0; j < width; j ++) {
            int col_idx = ((x_offset+j)<<x_subsampling) / palantir_cfg->patch_width;
            palantir_cfg->dependency_graph->block_residual[row_idx*palantir_cfg->num_patches_per_row+col_idx] += powf((float)buffer[(i+y_offset)*stride+j+x_offset]-(float)reconstructed[i*width+j], 2);
        }
    }

    vpx_free(padded_input);
    vpx_free(downscaled);
    vpx_free(reconstructed);
}

static void init_palantir_worker_data(palantir_worker_data_t *mwd, int index) {
    assert (mwd != NULL);

    mwd->lr_resiudal = (YV12_BUFFER_CONFIG *) vpx_calloc(1, sizeof(YV12_BUFFER_CONFIG));

    mwd->intra_block_list = (palantir_interp_block_list_t *) vpx_calloc(1, sizeof(palantir_interp_block_list_t));
    mwd->intra_block_list->cur = NULL;
    mwd->intra_block_list->head = NULL;
    mwd->intra_block_list->tail = NULL;

    mwd->inter_block_list = (palantir_interp_block_list_t *) vpx_calloc(1, sizeof(palantir_interp_block_list_t));
    mwd->inter_block_list->cur = NULL;
    mwd->inter_block_list->head = NULL;
    mwd->inter_block_list->tail = NULL;

    mwd->index = index;

    mwd->latency_log = NULL;
    mwd->metadata_log = NULL;
}

palantir_worker_data_t *init_palantir_worker(int num_threads, palantir_cfg_t *palantir_cfg) {
    char latency_log_path[PATH_MAX];
    char metadata_log_path[PATH_MAX];

    if (!palantir_cfg) {
        fprintf(stderr, "%s: palantir_cfg is NULL", __func__);
        return NULL;
    }
    if (num_threads <= 0) {
        fprintf(stderr, "%s: num_threads is equal or less than 0", __func__);
        return NULL;
    }

    palantir_worker_data_t *mwd = (palantir_worker_data_t *) vpx_malloc(sizeof(palantir_worker_data_t) * num_threads);
    int i;
    for (i = 0; i < num_threads; ++i) {
        init_palantir_worker_data(&mwd[i], i);

        if (palantir_cfg->save_latency == 1) {
            sprintf(latency_log_path, "%s/latency_thread%d%d.txt", palantir_cfg->log_dir, mwd[i].index, num_threads);
            if ((mwd[i].latency_log = fopen(latency_log_path, "w")) == NULL) {
                fprintf(stderr, "%s: cannot open a file %s", __func__, latency_log_path);
                palantir_cfg->save_latency = 0;
            }
        }

        if (palantir_cfg->save_metadata == 1) {
            sprintf(metadata_log_path, "%s/metadata_thread%d%d.txt", palantir_cfg->log_dir, mwd[i].index, num_threads);
            if ((mwd[i].metadata_log = fopen(metadata_log_path, "w")) == NULL) {
                fprintf(stderr, "%s: cannot open a file %s", __func__, metadata_log_path);
                palantir_cfg->save_metadata = 0;
            }
        }
    }

    return mwd;
}


palantir_bilinear_coeff_t *init_bilinear_coeff(int width, int height, int scale) {
    struct palantir_bilinear_coeff *coeff = (palantir_bilinear_coeff_t *) vpx_calloc(1, sizeof(palantir_bilinear_coeff_t));
    int x, y;

    assert (coeff != NULL);
    assert (width != 0 && height != 0 && scale > 0);

    coeff->x_lerp = (float *) vpx_malloc(sizeof(float) * width * scale);
    coeff->x_lerp_fixed = (int16_t *) vpx_malloc(sizeof(int16_t) * width * scale);
    coeff->left_x_index = (int *) vpx_malloc(sizeof(int) * width * scale);
    coeff->right_x_index = (int *) vpx_malloc(sizeof(int) * width * scale);

    coeff->y_lerp = (float *) vpx_malloc(sizeof(float) * height * scale);
    coeff->y_lerp_fixed = (int16_t *) vpx_malloc(sizeof(int16_t) * height * scale);
    coeff->top_y_index = (int *) vpx_malloc(sizeof(int) * height * scale);
    coeff->bottom_y_index = (int *) vpx_malloc(sizeof(int) * height * scale);

    for (x = 0; x < width * scale; ++x) {
        const double in_x = (x + 0.5f) / scale - 0.5f;
        coeff->left_x_index[x] = MAX(floor(in_x), 0);
        coeff->right_x_index[x] = MIN(ceil(in_x), width - 1);
        coeff->x_lerp[x] = in_x - floor(in_x);
        coeff->x_lerp_fixed[x] = coeff->x_lerp[x] * 32;
    }

    for (y = 0; y < height * scale; ++y) {
        const double in_y = (y + 0.5f) / scale - 0.5f;
        coeff->top_y_index[y] = MAX(floor(in_y), 0);
        coeff->bottom_y_index[y] = MIN(ceil(in_y), height - 1);
        coeff->y_lerp[y] = in_y - floor(in_y);
        coeff->y_lerp_fixed[y] = coeff->y_lerp[y] * 32;
    }

    return coeff;
}

void remove_bilinear_coeff(palantir_bilinear_coeff_t *coeff) {
    if (coeff != NULL) {
        vpx_free(coeff->x_lerp);
        vpx_free(coeff->x_lerp_fixed);
        vpx_free(coeff->left_x_index);
        vpx_free(coeff->right_x_index);

        vpx_free(coeff->y_lerp);
        vpx_free(coeff->y_lerp_fixed);
        vpx_free(coeff->top_y_index);
        vpx_free(coeff->bottom_y_index);

        vpx_free(coeff);
    }
}

void create_palantir_interp_block(palantir_interp_block_list_t *L, int mi_col, int mi_row, int n4_w,
                              int n4_h) {
    palantir_interp_block_t *newBlock = (palantir_interp_block_t *) vpx_calloc(1, sizeof(palantir_interp_block_t));
    newBlock->mi_col = mi_col;
    newBlock->mi_row = mi_row;
    newBlock->n4_w[0] = n4_w;
    newBlock->n4_h[0] = n4_h;
    newBlock->next = NULL;
    newBlock->prev = NULL;

    if (L->head == NULL && L->tail == NULL) {
        L->head = L->tail = newBlock;
    } else {
        newBlock->prev = L->tail;
        L->tail->next = newBlock;
        L->tail = newBlock;
    }

    L->cur = newBlock;
}

void set_palantir_interp_block(palantir_interp_block_list_t *L, int plane, int n4_w, int n4_h) {
    palantir_interp_block_t *currentBlock = L->cur;
    currentBlock->n4_w[plane] = n4_w;
    currentBlock->n4_h[plane] = n4_h;
}

void remove_palantir_interp_block_tail(palantir_interp_block_list_t *L) {
    if (L->tail==NULL) {
        return;
    }
    if (L->head == L->tail) {
        L->head = NULL;
    }
    L->cur = L->tail->prev;
    vpx_free(L->tail);
    L->tail = L->cur;
    if (L->tail!=NULL){
        L->tail->next = NULL;
    }
}

int RGB24_save_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path) {
    FILE *serialize_file = fopen(file_path, "wb");
    if (serialize_file == NULL) {
        fprintf(stderr, "%s: fail to save a file to %s\n", __func__, file_path);
        return -1;
    }

    uint8_t *src = rbf->buffer_alloc;
    int h = rbf->height;
    do {
        fwrite(src, sizeof(uint8_t), rbf->width, serialize_file);
        src += rbf->stride;
    } while (--h);

    fclose(serialize_file);

    return 0;
}

int RGB24_load_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path) {
    FILE *serialize_file = fopen(file_path, "rb");
    if (serialize_file == NULL) {
        fprintf(stderr, "%s: fail to open a file from %s\n", __func__, file_path);
        return -1;
    }

    uint8_t *src = rbf->buffer_alloc;
    int h = rbf->height;
    do {
        fread(src, sizeof(uint8_t), rbf->width, serialize_file);
        src += rbf->stride;
    } while (--h);

    fclose(serialize_file);

    return 0;
}

int RGB24_alloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height) {
    if (rbf) {
        RGB24_free_frame_buffer(rbf);
        return RGB24_realloc_frame_buffer(rbf, width, height);
    }
    return -1;
}

int RGB24_realloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height) {
    if (rbf) {
        const int stride = width * 3;

        const int frame_size = height * stride;

        if (frame_size > rbf->buffer_alloc_sz) {
            if (rbf->buffer_alloc_sz != 0) {
                vpx_free(rbf->buffer_alloc);
                vpx_free(rbf->buffer_alloc_float);
            }

            rbf->buffer_alloc = (uint8_t *) vpx_calloc(1, (size_t) frame_size * sizeof(uint8_t));
            if (!rbf->buffer_alloc) {
                return -1;
            }

            rbf->buffer_alloc_float = (float *) vpx_calloc(1, (size_t) frame_size * sizeof(float));
            if (!rbf->buffer_alloc_float) {
                return -1;
            }

            rbf->buffer_alloc_sz = (int) frame_size;
        }
        rbf->height = height;
        rbf->width = width * 3;
        rbf->stride = stride;

        return 0;
    }
    return -1;
}

int RGB24_free_frame_buffer(RGB24_BUFFER_CONFIG *rbf) {
    if (rbf) {
        if (rbf->buffer_alloc_sz > 0) {
            vpx_free(rbf->buffer_alloc);
            vpx_free(rbf->buffer_alloc_float);
        }
        memset(rbf, 0, sizeof(RGB24_BUFFER_CONFIG));
    } else {
        return -1;
    }
    return 0;
}

//from <vpx_dsp/src/psnr.c>
static void encoder_variance(const uint8_t *a, int a_stride, const uint8_t *b,
                             int b_stride, int w, int h, unsigned int *sse,
                             int *sum) {
    int i, j;

    *sum = 0;
    *sse = 0;

    for (i = 0; i < h; i++) {
        for (j = 0; j < w; j++) {
            const int diff = a[j] - b[j];
            *sum += diff;
            *sse += diff * diff;
        }

        a += a_stride;
        b += b_stride;
    }
}

//from <vpx_dsp/src/psnr.c>
static int64_t get_sse(const uint8_t *a, int a_stride, const uint8_t *b,
                       int b_stride, int width, int height) {
    const int dw = width % 16;
    const int dh = height % 16;
    int64_t total_sse = 0;
    unsigned int sse = 0;
    int sum = 0;
    int x, y;

    if (dw > 0) {
        encoder_variance(&a[width - dw], a_stride, &b[width - dw], b_stride, dw,
                         height, &sse, &sum);
        total_sse += sse;
    }

    if (dh > 0) {
        encoder_variance(&a[(height - dh) * a_stride], a_stride,
                         &b[(height - dh) * b_stride], b_stride, width - dw, dh,
                         &sse, &sum);
        total_sse += sse;
    }

    for (y = 0; y < height / 16; ++y) {
        const uint8_t *pa = a;
        const uint8_t *pb = b;
        for (x = 0; x < width / 16; ++x) {
            vpx_mse16x16(pa, a_stride, pb, b_stride, &sse);
            total_sse += sse;

            pa += 16;
            pb += 16;
        }

        a += 16 * a_stride;
        b += 16 * b_stride;
    }

    return total_sse;
}

double RGB24_calc_psnr(const RGB24_BUFFER_CONFIG *a, const RGB24_BUFFER_CONFIG *b) {
    static const double peak = 255.0;
    double psnr;

    const int w = a->width;
    const int h = a->height;
    const uint32_t samples = w * h;
    const uint64_t sse = get_sse(a->buffer_alloc, a->stride, b->buffer_alloc, b->stride, w, h);

    psnr = vpx_sse_to_psnr(samples, peak, (double) sse);

    return psnr;
}
