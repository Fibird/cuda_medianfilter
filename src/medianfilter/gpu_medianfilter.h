/**
 * Author: Liu Chaoyang
 * E-mail: chaoyanglius@gmail.com
 * 
 * Median filter using cuda C(shared memory)
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

#ifndef GPU_MEDIANFILTER_H
#define GPU_MEDIANFILTER_H

#include <opencv2/core/core.hpp>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

// Signal/image element type
typedef float element;

//   1D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     length - length of the signal
//     w_width - width of window
//     ts_per_bk - threads per block
void medianfilter1D(element* signal, element* result, unsigned length, int w_width, int ts_per_bk);

//   2D MEDIAN FILTER wrapper
//     signal - input signal
//     result - output signal
//     length - length of the signal
//     k_width - width of kernel 
//     ts_per_bk - threads per dimension of block
void medianfilter2D(const cv::Mat &src, cv::Mat &dst, int k_width, int ts_per_dm);
void test_test();
#endif
