#pragma once

#ifndef WARP_GPU_H
#define WARP_GPU_H
#include <Prerequisites.cuh>
#include "Types.h"
#include "liblionImports.h"

using namespace relion;

//Memory.cpp
extern "C" float* __stdcall MallocDeviceFromHost(float* h_data, long long elements);
extern "C" float* __stdcall MallocDevice(long long elements);
extern "C" void __stdcall CopyDeviceToHost(float* d_source, float* h_dest, long long elements);
extern "C" void __stdcall FreeDevice(void* d_data);
extern "C" void SubtractFromSlices(float* d_input, float* d_subtrahends, float* d_output, size_t sliceelements, uint slices);
extern "C"  void MultiplyByScalar(float* d_input, float* d_output, float multiplicator, size_t elements);
extern "C" void __stdcall CopyHostToDevice(float* h_source, float* d_dest, long long elements);

//Tools.cu
extern "C" void Scale(float* d_input, float* d_output, int3 dimsinput, int3 dimsoutput, uint batch, int planforw, int planback, float2* d_inputfft, float2* d_outputfft);
extern "C" void SphereMask(float* d_input, float* d_output, int3 dims, float radius, float sigma, bool decentered, uint batch);


//Wrappers
void ResizeMapGPU(MultidimArray<float> &img, int3 newDim);
void SphereMaskGPU(float* d_input, float* d_output, int3 dims, float radius, float sigma, bool decentered, uint batch);
void ResizeMapGPU(MultidimArray<float> &img, int2 newDim);
void Substract_GPU(MultidimArray<float> &img, float substrahend);
void Substract_GPU(MultidimArray<float> &img, MultidimArray<DOUBLE> &substrahend);
#endif