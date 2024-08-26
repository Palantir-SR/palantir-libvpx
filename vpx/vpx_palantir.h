//
// Created by hyunho on 9/2/19.
//

#ifndef LIBVPX_WRAPPER_VPX_SR_CACHE_H
#define LIBVPX_WRAPPER_VPX_SR_CACHE_H

#include <limits.h>
#include "./vpx_config.h"
#include "vpx_scale/yv12config.h"

/* Structure to log latency */
typedef struct palantir_latency {
    //decode
    double decode;
    double decode_intra_block;
    double decode_inter_block;
    double decode_inter_residual;

    //interpolation (interp)
    double interp;
    double interp_intra_block;
    double interp_inter_block;
    double interp_inter_residual;

    //super-resolution (sr)
    double sr;
    double sr_convert_rgb_to_yuv;
    double sr_execute_dnn;
    double sr_data_copy;
    double sr_convert_yuv_to_rgb;
    double sr_convert_float_to_int;
    int sr_num_anchors;
} palantir_latency_t;

/* Structure to log frame index */
typedef struct palantir_frame {
    int video_frame_index;
    int super_frame_index;
} palantir_frame_t;

/* Structure to log metdata */
typedef struct palantir_metadata {
    palantir_frame_t reference_frames[3];
    int num_blocks;
    int num_intrablocks;
    int num_interblocks;
    int num_noskip_interblocks;
} palantir_metdata_t;

/* Enum to set a decode mode */
typedef enum{
    DECODE,
    DECODE_SR,
    DECODE_CACHE,
    DECODE_BLOCK_CACHE
} palantir_decode_mode;

/* Enum to set a cache mode */
typedef enum{
    NO_CACHE,
    PROFILE_CACHE,
    KEY_FRAME_CACHE,
} palantir_cache_mode;

/* Enum to set a dnn mode */
typedef enum{
    NO_DNN,
    ONLINE_DNN,
    OFFLINE_DNN,
} palantir_dnn_mode;

/* Enum to set a DNN runtime */
typedef enum{
    CPU_FLOAT32,
    GPU_FLOAT32_16_HYBRID,
    DSP_FIXED8,
    GPU_FLOAT16,
    AIP_FIXED8
} palantir_dnn_runtime;

/* Structure to hold coefficients to run bilinear interpolation */
typedef struct palantir_bilinear_coeff{
    float *x_lerp;
    int16_t *x_lerp_fixed;
    float *y_lerp;
    int16_t *y_lerp_fixed;
    int *top_y_index;
    int *bottom_y_index;
    int *left_x_index;
    int *right_x_index;
} palantir_bilinear_coeff_t;


/* Structure for a RGB888 frame  */
typedef struct rgb24_buffer_config{
    int width;
    int height;
    int stride;
    int buffer_alloc_sz;
    uint8_t *buffer_alloc;
    float *buffer_alloc_float; //TODO: remove this
} RGB24_BUFFER_CONFIG;


/* Structure to read a cache profile */
//TODO: support file_size
typedef struct palantir_cache_profile {
    FILE *file;
    uint64_t offset; // offset in the profile file
    uint8_t offset_byte; // offset in byte_value
    uint8_t byte_value;
    off_t file_size;
    int num_dummy_bits;
    int *block_apply_dnn;
    RGB24_BUFFER_CONFIG *anchor_block_input_buffer;
    RGB24_BUFFER_CONFIG *anchor_block_sr_buffer;
} palantir_cache_profile_t;

/* Structure to read a cache profile */
typedef struct palantir_dependency_graph {
    float *block_residual;
    float *block_encoding_size;
    float *block_dqvs;
    float **block_dependency_weight;
} palantir_dependency_graph_t;

/* Structure to hold the location of block, which will be interpolated */
typedef struct palantir_interp_block{
    int mi_row;
    int mi_col;
    int n4_w[3];
    int n4_h[3];
    struct palantir_interp_block *next;
    struct palantir_interp_block *prev;
} palantir_interp_block_t;

/* Structure to hold a list of block locations */
typedef struct palantir_interp_block_list{
    palantir_interp_block_t *cur;
    palantir_interp_block_t *head;
    palantir_interp_block_t *tail;
} palantir_interp_block_list_t;

/* Structure to hold per-thread information  */
typedef struct palantir_worker_data {
    //interpolation
    YV12_BUFFER_CONFIG *lr_resiudal;
    palantir_interp_block_list_t *intra_block_list;
    palantir_interp_block_list_t *inter_block_list;

    //log
    int index;
    FILE *latency_log;
    FILE *metadata_log;
    palantir_latency_t latency;
    palantir_metdata_t metadata;
} palantir_worker_data_t;

typedef struct palantir_dnn{
    void *interpreter;
    int scale;
} palantir_dnn_t;

/* Struture to hold per-decoder information */
typedef struct palantir_cfg{
    //direcetory
    char log_dir[PATH_MAX];
    char input_frame_dir[PATH_MAX];
    char sr_frame_dir[PATH_MAX];
    char input_reference_frame_dir[PATH_MAX];
    char sr_reference_frame_dir[PATH_MAX];
    char sr_offline_frame_dir[PATH_MAX]; //OFFLINE_DNN (load images)
    char dnn_dir[PATH_MAX];
    char dnn_name[PATH_MAX];

    //logging
    int save_rgbframe; // rgb
    int save_yuvframe;
    int save_quality;
    int save_quality_block_format;
    int save_latency;
    int save_metadata;
    int save_frame_size;
    int save_finegrained_metadata;
    int save_super_finegrained_metadata;
    int filter_interval;

    //mode
    palantir_decode_mode decode_mode;
    palantir_cache_mode cache_mode;
    palantir_dnn_mode dnn_mode;
    palantir_dnn_runtime dnn_runtime;
    int target_width;
    int target_height;

    //profile
    palantir_dnn_t *dnn;
    palantir_cache_profile_t *cache_profile;
    palantir_dependency_graph_t *dependency_graph;
    palantir_bilinear_coeff_t *bilinear_coeff;
    int gop;
    int num_patches_per_row;
    int num_patches_per_column;
    int patch_width;
    int patch_height;
    int chunk_idx;

} palantir_cfg_t;


#ifdef __cplusplus
extern "C" {
#endif

palantir_cfg_t *init_palantir_cfg();
void remove_palantir_cfg(palantir_cfg_t *config);

palantir_dnn_t *init_palantir_dnn(int scale);
void remove_palantir_dnn(palantir_dnn_t *dnn);

palantir_cache_profile_t *init_palantir_cache_profile();
void remove_palantir_cache_profile(palantir_cache_profile_t *cache_profile);
int read_cache_profile(palantir_cache_profile_t *cache_profile, int num_patches_per_row, int num_patches_per_column, int save_finegrained_metadata, int save_super_finegrained_metadata);
int read_cache_profile_dummy_bits(palantir_cache_profile_t *cache_profile);

void update_palantir_dependency_graph_block_residual_uint8(palantir_cfg_t *palantir_cfg, uint8_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling);


void update_palantir_dependency_graph_block_residual_v2_uint8(palantir_cfg_t *palantir_cfg, uint8_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling);

void update_palantir_dependency_graph_block_residual_int16(palantir_cfg_t *palantir_cfg, int16_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling);

void update_palantir_dependency_graph_block_residual_v2_int16(palantir_cfg_t *palantir_cfg, int16_t *buffer, int stride, int x_offset, int y_offset, int width, int height, int x_subsampling, int y_subsampling);

palantir_bilinear_coeff_t *init_bilinear_coeff(int width, int height, int scale);
void remove_bilinear_coeff(palantir_bilinear_coeff_t *coeff);

void create_palantir_interp_block(struct palantir_interp_block_list *L, int mi_col, int mi_row, int n4_w,
                              int n4_h);
void set_palantir_interp_block(struct palantir_interp_block_list *L, int plane, int n4_w, int n4_h);
void remove_palantir_interp_block_tail(struct palantir_interp_block_list *L);

palantir_worker_data_t *init_palantir_worker(int num_threads, palantir_cfg_t *palantir_cfg);
void remove_palantir_worker(palantir_worker_data_t *mwd, int num_threads);

int RGB24_save_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path);
int RGB24_load_frame_buffer(RGB24_BUFFER_CONFIG *rbf, char *file_path);
int RGB24_alloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height);
int RGB24_realloc_frame_buffer(RGB24_BUFFER_CONFIG *rbf, int width, int height);
int RGB24_free_frame_buffer(RGB24_BUFFER_CONFIG *rbf);
double RGB24_calc_psnr(const RGB24_BUFFER_CONFIG *a, const RGB24_BUFFER_CONFIG *b);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif //LIBVPX_WRAPPER_VPX_SR_CACHE_H
