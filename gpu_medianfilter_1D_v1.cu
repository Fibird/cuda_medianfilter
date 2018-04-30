/**
 * Author: Liu Chaoyang
 * E-mail: chaoyanglius@gmail.com
 * 
 * Median filter using cuda C(global memory)
 * Copyright (C) 2018 Liu Chaoyang
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <memory.h>
#include "waveformat/waveformat.h"
#include <cuda_runtime.h>

#define WINDOW_WIDTH 5

// signal/image element type
typedef short element;

//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     length - length of the signal
__global__ void _medianfilter(const element* signal, element* result, int length)
{
	element window[WINDOW_WIDTH];
	int gindex = threadIdx.x + blockDim.x * blockIdx.x;
	int radius = WINDOW_WIDTH / 2;

	while (gindex < length)
	{
		for (int j = 0; j < 2 * radius + 1; ++j)
			window[j] = signal[gindex + j];
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
		gindex += blockDim.x * gridDim.x; 
	}
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
	//   Treat special case N = 1
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
    dim3 block(1024, 1);
    dim3 grid((length + block.x - 1) / block.x, 1);

	//   Call median filter implementation
	_medianfilter<<<grid, block>>>(dev_extension, dev_result, length);

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
    // get header info of input file
    waveFormat fmt = readWaveHeader(fp);
    int size = fmt.data_size;
    
    // allocate host memory for input and output data
    signal = (element *)malloc(size * sizeof(element));
    // move fp to the beginning position of data
    fseek(fp, 44L, SEEK_SET);     
    // read signal data from input file
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
	fp = fopen("audios/gpu_v1_rst.wav", "wb+");

	if (fp == NULL)
        printf("Open output file failed!\n");

    // write header info into output file
    writeWaveHeader(fmt, fp);
    // move fp to the beginning position of data
    fseek(fp, 44L, SEEK_SET);
    // write result data into output file
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
