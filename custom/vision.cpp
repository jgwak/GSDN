// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved. MIT License.
// Modified and redistributed by JunYoung Gwak
#include "nms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms", &nms, "non-maximum suppression");
}
