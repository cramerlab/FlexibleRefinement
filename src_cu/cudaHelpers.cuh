#pragma once
#ifndef CUDA_HELPERS
#define CUDA_HELPERS


#include<cuda.h>
#include<cufft.h>
#include<cuda_runtime_api.h>
#include<stdio.h>
#include "Prerequisites.cuh"

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define cufftErrchk(ans) { cufftAssert((ans), __FILE__, __LINE__); }
const char *_cudaGetErrorEnum(cufftResult error);
void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true);
void cufftAssert(cufftResult_t code, const char *file, int line, bool abort = true );
using namespace gtom;
void d_floatToComplex(tfloat* d_input, tcomplex* d_output, size_t elements);
void d_OwnIFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch, bool renormalize);
__device__ float d_Lerp(float a, float b, float x);

void d_MaxOp(float* d_input1, float input2, float* d_output, size_t elements);
void d_SphereMaskFT(tcomplex* d_input, tcomplex* d_output, int3 dims, int radius, uint batch);

#endif // !CUDA_HELPERS