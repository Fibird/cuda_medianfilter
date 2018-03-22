#include <stdio.h>
#include <memory.h>
#include <cuda_runtime.h>
#include "waveformat/waveformat.h"

#define WINDOW_WIDTH 9
#define THREADS_PER_BLOCK 1024

// Signal/image element type
typedef int element;
//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     length - length of the signal
__global__ void _medianfilter(const element* signal, element* result, int length)
{
	element window[WINDOW_WIDTH];
    int radius = WINDOW_WIDTH / 2;
	__shared__ element cache[THREADS_PER_BLOCK + 2 * (WINDOW_WIDTH / 2)];

	int gindex = threadIdx.x + blockDim.x * blockIdx.x;
	int lindex = threadIdx.x + radius;
	// Reads input elements into shared memory
	cache[lindex] = signal[gindex];
	if (threadIdx.x < radius)
	{
		cache[lindex - radius] = signal[gindex - radius];
		cache[lindex + THREADS_PER_BLOCK] = signal[gindex + THREADS_PER_BLOCK];
	}
	__syncthreads();
	for (int j = 0; j < 2 * radius + 1; ++j)
		window[j] = cache[threadIdx.x + j];
	// Orders elements (only half of them)
	for (int j = 0; j < radius + 1; ++j)
	{
		// Finds position of minimum element
		int min = j;
		for (int k = j + 1; k < 2 * radius + 1; ++k)
			if (window[k] < window[min])
				min = k;
		// Puts found minimum element in its place
		const element temp = window[j];
		window[j] = window[min];
		window[min] = temp;
	}
	// Gets result - the middle element
	result[gindex] = window[radius];
}

//   1D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     length - length of the signal
void medianfilter(element* signal, element* result, int length)
{
	element *dev_extension, *dev_result;
    int radius = WINDOW_WIDTH / 2;

	//   Check arguments
	if (!signal || length < 1)
		return;
	//   Treat special case length = 1
	if (length == 1)
	{
		if (result)
			result[0] = signal[0];
		return;
	}
	//   Allocate memory for signal extension
	element* extension = (element*)malloc((length + 2 * radius) * sizeof(element));
	//   Check memory allocation
	if (!extension)
		return;
	//   Create signal extension
	cudaMemcpy(extension + 2, signal, length * sizeof(element), cudaMemcpyHostToHost);
	for (int i = 0; i < radius; ++i)
	{
		extension[i] = signal[1 - i];
		extension[length + radius + i] = signal[length - 1 - i];
	}

	cudaMalloc((void**)&dev_extension, (length + 2 * radius) * sizeof(int));
	cudaMalloc((void**)&dev_result, length * sizeof(int));

	// Copies signal to device
	cudaMemcpy(dev_extension, extension, (length + 2 * radius) * sizeof(element), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 block(THREADS_PER_BLOCK, 1);
    dim3 grid((length + block.x - 1) / block.x, 1);

	//   Call median filter implementation
	_medianfilter<<<grid, block>>>(dev_extension + radius, dev_result, length);
	// Copies result to host
	cudaMemcpy(result, dev_result, length * sizeof(element), cudaMemcpyDeviceToHost);

	// Free memory
	free(extension);
	cudaFree(dev_extension);
	cudaFree(dev_result);
}

int main(int argc, char **argv)
{
	element *signal, *result;
    
    if (argc != 2)
    {
        printf("Please specify name of file!\n");
        exit(EXIT_FAILURE);
    }

	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // read input music file
	FILE *fp;
    fp = fopen(argv[1], "rb");
    if (!fp)
    {
        printf("open input file failed!\n");
        return -1;
    }
    // get info of file
    waveFormat fmt = readWaveHeader(fp);
    int size = fmt.data_size;
    
    // allocate host memory for input and output data
    signal = (element *)malloc(size * sizeof(element));
    // get data of input file
    fseek(fp, 44L, SEEK_SET);
    fread(signal, sizeof(short), size, fp);
    // close file stream
    if (fp)
    {
        fclose(fp);
        fp = NULL;
    }

    // allocate host memory for output data
	result = (element *)malloc(size * sizeof(element));

    // execute median filter and time it
	cudaEventRecord(start, 0);
	medianfilter(signal, result, size);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%.3lf ms\n", elapsedTime);

    // save output data
	fp = fopen("audios/gpu_v2_rst.wav", "wb+");

	if (fp == NULL)
        printf("Open output file failed!\n");

    writeWaveHeader(fmt, fp);
    fseek(fp, 44L, SEEK_SET);
    fwrite(result, sizeof(short), size, fp);
    
    // close file stream
    if (fp)
    {
	    fclose(fp);
        fp = NULL;
    }

    // free host memory
    free(signal);
    free(result);

	return 0;
}
