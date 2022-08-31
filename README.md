## Introduction

CVFusion is an open-source deep learning compiler to fuse the OpenCV operators in the pre-processing stage of deep learning models, which can be used in [MMDeploy](https://github.com/open-mmlab/mmdeploy).

## Requirements

- A recent c++ compiler supporting C++ 14 (g++-5 or higher)
- CMake 3.8 or higher
- CUDA 8.0 or higher
## Installation

```shell
git clone git@github.com:OpenComputeLab/CVFusion.git
mkdir build && cd build
cmake ..
make -j8
```


## Usage

### For MMDeploy

#### Auto code-gen the image preprocessing operators written in C++

```shell
cd build/examples/MMDeploy
./OpFuse <path/of/OpList/json/file> <cpu or cuda> <path/of/generate/code>
```

#### Fuse function interface(cpu)

```
void FuseKernel(uint64_t resize_h, uint64_t resize_w, uint64_t crop_h, uint64_t crop_w, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest");
// if no ResizeOp, use default "nearest" to replace
```

#### Fuse function interface(cuda)

```
void FuseKernelCU(cudaStream_t stream, uint64_t resize_h, uint64_t resize_w, uint64_t crop_h, uint64_t crop_w, int32_t crop_top, int32_t crop_left, float norm_mean_0, float norm_mean_1, float norm_mean_2, float norm_std_0, float norm_std_1, float norm_std_2, uint64_t pad_h, uint64_t pad_w, int32_t pad_top, int32_t pad_left, int32_t pad_bottom, int32_t pad_right, float pad_value, uint8_t* __restrict__ src_raw_data, float* __restrict__ dst_raw_data, uint64_t dst_h, uint64_t dst_w, uint64_t src_h, uint64_t src_w, const char *format, const char *interpolation = "nearest");
// Note that there are there additional parameters: stream, dst_h and dst_w (Length and width of the resulting image)
```
