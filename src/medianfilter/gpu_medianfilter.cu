#include <memory.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include "gpu_medianfilter.h"

__global__ void _medianfilter1D(const element* signal, element* result, unsigned length, int w_width, int ts_per_bk)
{
	element *window = (element*)malloc(sizeof(element) * w_width);
    int radius = w_width / 2;
    extern __shared__ element cache[];

	int gindex = threadIdx.x + blockDim.x * blockIdx.x;
	int lindex = threadIdx.x + radius;
	// Reads input elements into shared memory
	cache[lindex] = signal[gindex];
	if (threadIdx.x < radius)
	{
		cache[lindex - radius] = signal[gindex - radius];
		cache[lindex + ts_per_bk] = signal[gindex + ts_per_bk];
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
    free(window);
}

void medianfilter1D(element* signal, element* result, unsigned length, int w_width, int ts_per_bk)
{
	element *dev_extension, *dev_result;
    int radius = w_width / 2;

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
    // Marginal elements
	for (int i = 0; i < radius; ++i)
	{
		extension[i] = signal[radius - i - 1];
		extension[length + radius + i] = signal[length - 1 - i];
	}

	cudaMalloc((void**)&dev_extension, (length + 2 * radius) * sizeof(element));
	cudaMalloc((void**)&dev_result, length * sizeof(element));

	// Copies signal to device
	cudaMemcpy(dev_extension, extension, (length + 2 * radius) * sizeof(element), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 block(ts_per_bk, 1);
    dim3 grid((length + block.x - 1) / block.x, 1);
    unsigned shared_size = ts_per_bk + 2 * (w_width / 2);

	//   Call median filter implementation
	_medianfilter1D<<<grid, block, shared_size>>>(dev_extension + radius, dev_result, length, w_width, ts_per_bk);
	// Copies result to host
	cudaMemcpy(result, dev_result, length * sizeof(element), cudaMemcpyDeviceToHost);

	// Free memory
	free(extension);
	cudaFree(dev_extension);
	cudaFree(dev_result);
}

__global__ void _medianfilter2D(const element* signal, element* result, unsigned width, unsigned height, int k_width, int ts_per_dm)
{
	element *kernel = (element*)malloc(sizeof(element) * k_width * k_width);
    int radius = k_width / 2;
    // use dynamic size shared memory
    extern __shared__ element cache[];
    int sh_cols = ts_per_dm + radius * 2;
    int sh_rows = ts_per_dm + radius * 2;
    //int bk_cols = ts_per_dm;    int bk_rows = ts_per_dm;
    unsigned sg_cols = width + radius * 2;
    unsigned sg_rows = height + radius * 2;

	int gl_ix = threadIdx.x + blockDim.x * blockIdx.x;
	int gl_iy = threadIdx.y + blockDim.y * blockIdx.y;
    int ll_ix = threadIdx.x + radius;
    int ll_iy = threadIdx.y + radius;

	// Reads input elements into shared memory
	cache[ll_iy * sh_cols + ll_ix] = signal[gl_iy * sg_cols + gl_ix];
	if (threadIdx.x < radius)
	{
        for (int i = 0; i < radius; i++)
        {
            cache[ll_iy * sh_cols + ll_ix - radius + i] = signal[gl_iy * sg_cols + gl_ix + radius - 1 - i];
            cache[ll_iy * sh_cols + sh_cols + ll_ix - i - 1] = signal[gl_iy * sg_cols + sg_cols + gl_ix - radius + i];
        }
	}
	if (threadIdx.y < radius)
	{
        for (int i = 0; i < radius; i++)
        {
            cache[ll_ix * sh_rows + ll_iy - radius + i] = signal[gl_ix * sg_rows + gl_iy + radius - 1 - i];
            cache[ll_ix * sh_rows + sh_rows + ll_iy - i - 1] = signal[gl_ix * sg_rows + sg_rows + gl_iy - radius + i];
        }
	}
	__syncthreads();

    // Get kernel element 
    for (int i = 0; i < k_width; ++i)
	    for (int j = 0; j < k_width * k_width; ++j)
	    	kernel[j] = cache[(threadIdx.y + i) * k_width + threadIdx.x + j];

	// Orders elements (only half of them)
	for (int j = 0; j < k_width * k_width / 2 + 1; ++j)
	{
		// Finds position of minimum element
		int min = j;
		for (int k = j + 1; k < 2 * radius + 1; ++k)
			if (kernel[k] < kernel[min])
				min = k;
		// Puts found minimum element in its place
		const element temp = kernel[j];
		kernel[j] = kernel[min];
		kernel[min] = temp;
	}
	// Gets result - the middle element
	result[gl_iy * sg_cols + gl_ix] = kernel[k_width * k_width / 2];
    free(kernel);
}

void medianfilter2D(const cv::Mat &src, cv::Mat &dst, int k_width, int ts_per_dm)
{
    unsigned width = src.size().width;
    unsigned height = src.size().height;

	//   Check arguments
	if (!src.data || width < 1 || height < 1)
		return;
    if (!src.isContinuous())
        return; 

	//   Treat special case length = 1
	if (width == 1 && height == 1)
	{
        dst = src;
		return;
	}

	element *dev_extension, *dev_result;
    element *extension, *result;
    int radius = k_width / 2;

	/////   Allocate page-locked memory for image extension 
	cudaMallocHost((void**)&extension, (width + 2 * radius) * (height + 2 * radius) * sizeof(element));
    cudaMallocHost((void**)&result, width * height * sizeof(element));
	//   Check memory allocation
	if (!extension)
		return;

	/////   Create image extension
    // Inner elements
    for (unsigned i = 0; i < height; ++i)
        cudaMemcpy(extension + (width + radius + radius) * (i + radius) +  radius, src.data + width * i, width * sizeof(element), cudaMemcpyHostToHost);
    // marginal elements
    for (int i = 0; i < radius; ++i)
    {
        cudaMemcpy(extension + (width + radius + radius) * (radius - i - 1), src.data + width * i, width * sizeof(element), cudaMemcpyHostToHost);
        cudaMemcpy(extension + (width + radius + radius) * (height + radius + i), src.data + width * (height - i - 1), width * sizeof(element), cudaMemcpyHostToHost); 
    }
	for (int i = 0; i < height; ++i)
	{
        for (int j = 0; j < radius; ++j)
        {
		    extension[(width + radius + radius) * (radius + i) + j] = src.data[width * i + radius - 1  - j];
		    extension[(width + radius + radius) * (radius + i) + width + radius + j] = src.data[width * i + width - (radius - j - 1)];
        }
	}

    // Allocate device memory
	cudaMalloc((void**)&dev_extension, (width + 2 * radius) * (height + 2 * radius) * sizeof(element));
	cudaMalloc((void**)&dev_result, width * height * sizeof(element));

	// Copies extension to device
	cudaMemcpy(dev_extension, extension, (width + 2 * radius) * (height + 2 * radius) * sizeof(element), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 block(ts_per_dm, ts_per_dm);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    unsigned shared_size = (ts_per_dm + 2 * radius) * (ts_per_dm + 2 * radius);

	//   Call median filter implementation
	_medianfilter2D<<<grid, block, shared_size>>>(dev_extension + radius, dev_result, width, height, k_width, ts_per_dm);
	// Copies result to host
	cudaMemcpy(result, dev_result, width * height * sizeof(element), cudaMemcpyDeviceToHost);

    element *dstData = (element*)malloc(width * height * sizeof(element));
    cudaMemcpy(dstData, result, width * height * sizeof(element), cudaMemcpyHostToHost);

    // Create dst image
    dst = cv::Mat(height, width, src.type(), dstData);

	// Free memory
	cudaFreeHost(extension);
	cudaFree(dev_extension);
	cudaFree(dev_result);
}
