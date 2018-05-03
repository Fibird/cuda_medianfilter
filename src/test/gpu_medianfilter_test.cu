#include <stdio.h>
#include "gpu_medianfilter.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Please specify name of file!\n");
        exit(EXIT_FAILURE);
    }
    Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32FC1);
    Mat result;
    
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // execute median filter and time it
	cudaEventRecord(start, 0);

	medianfilter2D(image, result, 3, 32);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("%.3lf ms\n", elapsedTime);
    result.convertTo(result, CV_8UC1);
    imwrite("result/restor.jpg", result);
	return 0;
}

