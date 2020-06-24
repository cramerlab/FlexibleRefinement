#pragma once
#ifndef CUDA_HELPERS
#define CUDA_HELPERS


#include<cuda.h>
#include<cufft.h>
#include<cuda_runtime_api.h>
#include<stdio.h>

#define cudaErrchk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
#define cufftErrchk(ans) { cufftAssert((ans), __FILE__, __LINE__); }
const char *_cudaGetErrorEnum(cufftResult error);
void cudaAssert(cudaError_t code, const char *file, int line, bool abort = true);
void cufftAssert(cufftResult_t code, const char *file, int line, bool abort = true );


__device__ float d_Lerp(float a, float b, float x);
#endif // !CUDA_HELPERS