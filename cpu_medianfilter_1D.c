#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include "waveformat/waveformat.c"

#define WINDOW_WIDTH 5

// Signal/image element type
typedef short element;
//   1D MEDIAN FILTER implementation
//     signal - input signal
//     result - output signal
//     length - length of the signal
void _medianfilter(const element* signal, element* result, int length)
{
    int radius = WINDOW_WIDTH / 2;

	//   Move window through all elements of the signal
	for (int i = 2; i < length - radius; ++i)
	{
		//   Pick up window elements
		element window[2 * radius + 1];
		for (int j = 0; j < 2 * radius + 1; ++j)
			window[j] = signal[i - radius + j];
		//   Order elements (only half of them)
		for (int j = 0; j < radius + 1; ++j)
		{
			//   Find position of minimum element
			int min = j;
			for (int k = j + 1; k < 2 * radius + 1; ++k)
				if (window[k] < window[min])
					min = k;
			//   Put found minimum element in its place
			const element temp = window[j];
			window[j] = window[min];
			window[min] = temp;
		}
		//   Get result - the middle element
		result[i - radius] = window[radius];
	}
}

//   1D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     length - length of the signal
void medianfilter(element* signal, element* result, int length)
{
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
	element* extension = new element[length + 2 * radius];
	//   Check memory allocation
	if (!extension)
		return;
	//   Create signal extension
	memcpy(extension + radius, signal, length * sizeof(element));
	for (int i = 0; i < radius; ++i)
	{
		extension[i] = signal[1 - i];
		extension[length + radius + i] = signal[length - 1 - i];
	}
	//   Call median filter implementation
	_medianfilter(extension, result ? result : signal, length + 2 * radius);
	//   Free memory
	delete[] extension;
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
	clock_t start, stop;

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

	start = clock();
	medianfilter(signal, result, size);
	stop = clock();
	
	elapsedTime = 1000 * ((float) (stop - start)) / CLOCKS_PER_SEC;
	
	printf("%.3lf ms\n", elapsedTime);
    // save output data
	fp = fopen("audios/cpu_rst.wav", "wb+");

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
