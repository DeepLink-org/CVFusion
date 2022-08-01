#include <string>

namespace Runtime {

/* ----------------- BilinearResize Preprocess ------------------ */
/* for uint8_t input */
static constexpr const char *cpu_bilinear_preprocess_func = R"(

extern "C" void bilinear_resize_preprocess(uint64_t src_h, uint64_t src_w, uint64_t dst_h, uint64_t dst_w,
                       int16_t* __restrict__ cubfh, int16_t* __restrict__ cubfw,
                       int32_t* __restrict__ inth, int32_t* __restrict__ intw) {
  float scale_h = double(src_h) / dst_h;
  float scale_w = double(src_w) / dst_w;

  for (int j = 0; j < dst_h; ++j) {
    float fh = (float)((j + 0.5) * scale_h - 0.5f);
    int sh = floor(fh);
    fh -= sh;
    if (sh < 0) {
      fh = 0;
      sh = 0;
    }
    if (sh >= src_h) {
      fh = 0;
      sh = src_h - 1;
    }

    int int_h1 = INCREASE(sh, src_h);

    fh = fh * 2048;
    cubfh[j] = rint(2048 - fh);
    cubfh[dst_h + j] = rint(fh);

    inth[j] = sh;
    inth[dst_h + j] = int_h1;
  }

  for (int i = 0; i < dst_w; ++i) {
    float fw = (float)((i + 0.5) * scale_w - 0.5f);
    int sw = floor(fw);
    fw -= sw;

    if (sw < 0) {
      fw = 0;
      sw = 0;
    }
    if (sw >= src_w) {
      fw = 0;
      sw = src_w - 1;
    }
    int int_w1 = INCREASE(sw, src_w);
    fw = fw * 2048;
    cubfw[i] = rint(2048 - rint(fw));
    cubfw[dst_w + i] = rint(fw);

    intw[i] = sw;
    intw[dst_w + i] = int_w1;
  }
}

)";

/* for float input */
static constexpr const char *cpu_float_bilinear_preprocess_func = R"(

extern "C" void bilinear_float_resize_preprocess(uint64_t src_h, uint64_t src_w, uint64_t dst_h, uint64_t dst_w,
                       float* __restrict__ cubfh, float* __restrict__ cubfw,
                       int32_t* __restrict__ inth, int32_t* __restrict__ intw) {
  float scale_h = double(src_h) / dst_h;
  float scale_w = double(src_w) / dst_w;

  for (int j = 0; j < dst_h; ++j) {
    float fh = (float)((j + 0.5) * scale_h - 0.5f);
    int sh = floor(fh);
    fh -= sh;
    if (sh < 0) {
      fh = 0;
      sh = 0;
    }
    if (sh >= src_h) {
      fh = 0;
      sh = src_h - 1;
    }

    int int_h1 = INCREASE(sh, src_h);

    cubfh[j] = 1.0 - fh;
    cubfh[dst_h + j] = fh;

    inth[j] = sh;
    inth[dst_h + j] = int_h1;
  }

  for (int i = 0; i < dst_w; ++i) {
    float fw = (float)((i + 0.5) * scale_w - 0.5f);
    int sw = floor(fw);
    fw -= sw;

    if (sw < 0) {
      fw = 0;
      sw = 0;
    }
    if (sw >= src_w) {
      fw = 0;
      sw = src_w - 1;
    }
    int int_w1 = INCREASE(sw, src_w);

    cubfw[i] = 1.0 - fw;
    cubfw[dst_w + i] = fw;

    intw[i] = sw;
    intw[dst_w + i] = int_w1;
  }
}

)";

static constexpr const char *cuda_bilinear_preprocess_func = R"(

extern "C" __device__ void bilinear_resize_preprocess(uint64_t src_h, uint64_t dst_h, uint64_t src_w, uint64_t dst_w, 
    int16_t* __restrict__ cubfh, int32_t* __restrict__ inth, int16_t* __restrict__ cubfw, int32_t* __restrict__ intw) {

    int element_h = blockIdx.y * blockDim.y + threadIdx.y;
    int element_w = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_h >= dst_h || element_w >= dst_w) {
        return;
    }

    
    float scale_w = double(src_w) / dst_w;
    float scale_h = double(src_h) / dst_h;

    float fw = (float)((element_w + 0.5) * scale_w - 0.5f);
    float fh = (float)((element_h + 0.5) * scale_h - 0.5f);

    int sw = floor(fw);
    int sh = floor(fh);

    fw -= sw;
    if (sw < 0) {
      fw = 0;
      sw = 0;
    }
    if (sw >= src_w) {
      fw = 0;
      sw = src_w - 1;
    }

    int int_w1 = INCREASE(sw, src_w);

    fw = fw * 2048;
    cubfw[0] = rint(2048 - fw);
    cubfw[1] = rint(fw);

    intw[0] = sw;
    intw[1] = int_w1;
    
    fh -= sh;
    if (sh < 0) {
    fh = 0;
    sh = 0;
    }
    if (sh >= src_h) {
    fh = 0;
    sh = src_h - 1;
    }

    int int_h1 = INCREASE(sh, src_h);

    fh = fh * 2048;
    cubfh[0] = rint(2048 - fh);
    cubfh[1] = rint(fh);

    inth[0] = sh;
    inth[1] = int_h1;
}

)";

static constexpr const char *cuda_float_bilinear_preprocess_func = R"(

extern "C" __device__ void bilinear_float_resize_preprocess(uint64_t src_h, uint64_t dst_h, uint64_t src_w, uint64_t dst_w, 
    float* __restrict__ cubfh, int32_t* __restrict__ inth, float* __restrict__ cubfw, int32_t* __restrict__ intw) {
    int element_h = blockIdx.y * blockDim.y + threadIdx.y;
    int element_w = blockIdx.x * blockDim.x + threadIdx.x;
    if (element_h >= dst_h || element_w >= dst_w) {
        return;
    }
    
    float scale_w = double(src_w) / dst_w;
    float scale_h = double(src_h) / dst_h;

    float fw = (float)((element_w + 0.5) * scale_w - 0.5f);
    float fh = (float)((element_h + 0.5) * scale_h - 0.5f);

    int sw = floor(fw);
    int sh = floor(fh);

    fw -= sw;
    if (sw < 0) {
      fw = 0;
      sw = 0;
    }
    if (sw >= src_w) {
      fw = 0;
      sw = src_w - 1;
    }

    int int_w1 = INCREASE(sw, src_w);

    cubfw[0] = 1.0 - fw;
    cubfw[1] = fw;

    intw[0] = sw;
    intw[1] = int_w1;
    
    fh -= sh;
    if (sh < 0) {
    fh = 0;
    sh = 0;
    }
    if (sh >= src_h) {
    fh = 0;
    sh = src_h - 1;
    }

    int int_h1 = INCREASE(sh, src_h);

    cubfh[0] = 1.0 - fh;
    cubfh[1] = fh;

    inth[0] = sh;
    inth[1] = int_h1;
}

)";

// static constexpr const char *cuda_float_bilinear_preprocess_func = R"(



/* ----------------- Prelude ------------------ */
static constexpr const char *prelude = R"(
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define EQUAL(a,b) (strcmp((a),(b))==0)
#define INCREASE(x, l) ((x + 1) >= (l) ? (x) : ((x) + 1))
#define ABORT(Msg)                                                   \
  {                                                                        \
    std::cerr << ": \033[1;91m"                                            \
              << "[Fatal]"                                                 \
              << "\033[m " << __FILE__ << ": " << __FUNCTION__ << ": Line" \
              << __LINE__ << ": " << Msg << std::endl;                     \
    std::abort();                                                               \
  }

#include "elena_int.h"

)";

static constexpr const char *cuda_prelude = R"(
#include <cuda_runtime.h>
#define cuErrCheck(res)                                        \
    {                                                          \
        if (res != cudaSuccess)                                \
            ABORT("cuda assert: " << cudaGetErrorString(res)); \
    }
#define BLOCK_SIZE )";

/* ----------------- Call Function ------------------ */
/* for cpu */
static constexpr const char *cpu_call_func_begin = R"(

extern "C" void FuseKernel(uint64_t resize_h, uint64_t resize_w, uint64_t crop_h, uint64_t crop_w, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest"){
    if (resize_h && resize_w && EQUAL(interpolation, "nearest")) {
        if(EQUAL(format, "BGR")){
          BGR_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Nearest_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }
    } )";

static constexpr const char *cpu_bilinear_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        int* inth;
        int* intw;

        if(EQUAL(format, "NV12") || EQUAL(format, "NV21")) {
          float* cubfh = new float[resize_h*2];
          float* cubfw = new float[resize_w*2];
          inth = new int[resize_h*2];
          intw = new int[resize_w*2];

          bilinear_float_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

          if(EQUAL(format, "NV12"))
            NV12_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          else 
            NV21_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          
          if (cubfh) delete[] cubfh;
          if (cubfw) delete[] cubfw;
        } else {
          short* cubfh = new short[resize_h*2];
          short* cubfw = new short[resize_w*2];
          inth = new int[resize_h*2];
          intw = new int[resize_w*2];

          bilinear_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

          if(EQUAL(format, "BGR")){
            BGR_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else if(EQUAL(format, "RGB")){
            RGB_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else if(EQUAL(format, "GRAY")){
            GRAY_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else if(EQUAL(format, "BGRA")){
            BGRA_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
          } else {
            ABORT("This format is not supported");
          }

          if (cubfh) delete[] cubfh;
          if (cubfw) delete[] cubfw;
        }

        if (inth) delete[] inth;
        if (intw) delete[] intw;
    })";

static constexpr const char *cpu_bilinear_float_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){
        int* inth;
        int* intw;
        float* cubfh;
        float* cubfw;

        cubfh = new float[resize_h*2];
        cubfw = new float[resize_w*2];
        inth = new int[resize_h*2];
        intw = new int[resize_w*2];

        bilinear_float_resize_preprocess(src_h, src_w, resize_h, resize_w, cubfh, cubfw, inth, intw);

        if(EQUAL(format, "BGR")){
          BGR_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Bilinear_Kernel(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, inth, intw, cubfh, cubfw, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
          ABORT("This format is not supported");
        }
        
        if (cubfh) delete[] cubfh;
        if (cubfw) delete[] cubfw;
        if (inth) delete[] inth;
        if (intw) delete[] intw;
    })";

/* for cuda */
static constexpr const char *cuda_call_func_begin = R"(

extern "C" void FuseKernelCU(cudaStream_t stream, uint64_t resize_h, uint64_t resize_w, uint64_t crop_h, uint64_t crop_w, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t dst_h, uint64_t dst_w, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest"){

    if (resize_h && resize_w && EQUAL(interpolation, "nearest")) {
        if(EQUAL(format, "BGR")){
          BGR_Nearest_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "RGB")){
          RGB_Nearest_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "GRAY")){
          GRAY_Nearest_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "BGRA")){
          BGRA_Nearest_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV12")){
          NV12_Nearest_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else if(EQUAL(format, "NV21")){
          NV21_Nearest_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
        } else {
           ABORT("This format is not supported");
        }
    } )";
  
static constexpr const char *cuda_bilinear_func = R"(
    else if(resize_h && resize_w && EQUAL(interpolation, "bilinear")){  
      if(EQUAL(format, "BGR")){
          BGR_Bilinear_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
      } else if(EQUAL(format, "RGB")){
          RGB_Bilinear_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
      } else if(EQUAL(format, "GRAY")){
          GRAY_Bilinear_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
      } else if(EQUAL(format, "BGRA")){
          BGRA_Bilinear_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
      } else if(EQUAL(format, "NV12")){
          NV12_Bilinear_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
      } else if(EQUAL(format, "NV21")){
          NV21_Bilinear_Kernel<<<dim3((dst_w+BLOCK_SIZE-1)/BLOCK_SIZE, (dst_h+BLOCK_SIZE-1)/BLOCK_SIZE,1), dim3(BLOCK_SIZE,BLOCK_SIZE,1), 0, stream>>>(resize_h, resize_w, crop_h, crop_w, crop_top, crop_left, norm_mean_0, norm_mean_1, norm_mean_2, norm_std_0, norm_std_1, norm_std_2, pad_h, pad_w, pad_top, pad_left, pad_bottom, pad_right, pad_value, src_raw_data, dst_raw_data, src_h, src_w);
      } else {
          ABORT("This format is not supported");
      }
    })";

/* Common func end */
static constexpr const char *call_func_end = R"(
    else {
      ABORT("This interpolation is not supported");
    }
}
)";
}  // namespace Runtime