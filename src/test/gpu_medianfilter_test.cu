#include <stdio.h>
#include "waveformat.h"

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
	fp = fopen("audios/gpu_v2_rst.wav", "wb+");

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

