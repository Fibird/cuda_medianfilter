#include <stdlib.h>
#include <memory.h>
#include "waveformat.h"
#include "cpu_medianfilter.h"

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
	element* extension = (element*)malloc((length + 2 * radius) * sizeof(element));
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
   free(extension);
}
