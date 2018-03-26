# Overview

Using cuda C to implement medianfilter.

# Build

## GPU

```
nvcc -arch=sm_xx gpu_medianfilter_1D_vx.cu waveformat/waveformat.c -o bin/gpu_vx
```

## CPU

```
gcc cpu_medianfilter_1D.c -o bin/cpu_exe
```

# Run

```
./bin/gpu_vx audios/moz_noisy.wav
```


