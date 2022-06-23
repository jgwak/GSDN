// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// Modified and redistributed by JunYoung Gwak
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float xmin = max(a[0], b[0]), xmax = min(a[3], b[3]);
  float ymin = max(a[1], b[1]), ymax = min(a[4], b[4]);
  float zmin = max(a[2], b[2]), zmax = min(a[5], b[5]);
  float xsize = max(xmax - xmin, 0.f), ysize = max(ymax - ymin, 0.f);
  float zsize = max(zmax - zmin, 0.f);
  float interS = xsize * ysize * zsize;
  float Sa = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2]);
  float Sb = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]);
  return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 7];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 7 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 0];
    block_boxes[threadIdx.x * 7 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 1];
    block_boxes[threadIdx.x * 7 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 2];
    block_boxes[threadIdx.x * 7 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 3];
    block_boxes[threadIdx.x * 7 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 4];
    block_boxes[threadIdx.x * 7 + 5] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 5];
    block_boxes[threadIdx.x * 7 + 6] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 7 + 6];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 7;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 7) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = THCCeilDiv(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

// boxes is a N x 7 tensor
at::Tensor nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {
  using scalar_t = float;
  AT_ASSERTM(boxes.type().is_cuda(), "boxes must be a CUDA tensor");
  auto scores = boxes.select(1, 6);
  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));
  auto boxes_sorted = boxes.index_select(0, order_t);

  int boxes_num = boxes.size(0);

  const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

  scalar_t* boxes_dev = boxes_sorted.data<scalar_t>();

  THCState *state = at::globalContext().lazyInitCUDA(); // TODO replace with getTHCState

  unsigned long long* mask_dev = NULL;
  //THCudaCheck(THCudaMalloc(state, (void**) &mask_dev,
  //                      boxes_num * col_blocks * sizeof(unsigned long long)));

  mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

  dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
              THCCeilDiv(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  nms_kernel<<<blocks, threads>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  THCudaCheck(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);
  std::vector<unsigned long long> remi(boxes_num);
  memset(&remi[0], -1, sizeof(unsigned long long) * boxes_num);

  at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
  int64_t* keep_out = keep.data<int64_t>();

  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[i] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        unsigned long long is_new_overlap = p[j] & ~remv[j];
        int start_thread;
        if (j == nblock) {
            start_thread = inblock + 1;
        } else {
            start_thread = 0;
        }
        for (int k = start_thread; k < threadsPerBlock; k++) {
            if(is_new_overlap & (1ULL << k)) {
                remi[j * threadsPerBlock + k] = i;
            }
        }
        remv[j] |= p[j];
      }
    } else {
        keep_out[i] = remi[i];
    }
  }

  THCudaFree(state, mask_dev);
  return order_t.index({keep.to(order_t.device(), keep.scalar_type())});
}
