# Overview

Using cuda C to implement medianfilter.

# Compile

```
nvcc -arch=sm_xx gpu_medianfilter_1D_v1.cu waveformat/waveformat.c -o bin/gpu_vx
```

Note:sm_xx can be sm_30, sm_35 or sm_60 .... 
