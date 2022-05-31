#pragma once

#ifndef WARP_GPU_H
#define WARP_GPU_H
#include <Prerequisites.cuh>
#include "Types.h"
#include "liblionImports.h"

using namespace relion;

//Angles.cpp

extern "C" int __stdcall GetAnglesCount(int healpixorder, char* c_symmetry, float limittilt);
extern "C" void __stdcall GetAngles(float3* h_angles, int healpixorder, char* c_symmetry, float limittilt);

//Memory.cpp
extern "C" float* __stdcall MallocDeviceFromHost(float* h_data, long long elements);
extern "C" float* __stdcall MallocDevice(long long elements);
extern "C" void __stdcall CopyDeviceToHost(float* d_source, float* h_dest, long long elements);
extern "C" void __stdcall FreeDevice(void* d_data);
extern "C" void SubtractFromSlices(float* d_input, float* d_subtrahends, float* d_output, size_t sliceelements, uint slices);
extern "C"  void MultiplyByScalar(float* d_input, float* d_output, float multiplicator, size_t elements);
extern "C" void __stdcall CopyHostToDevice(float* h_source, float* d_dest, long long elements);

//Tools.cu
extern "C" void __stdcall ProjectBackward(float2* d_volumeft, float* d_volumeweights, int3 dimsvolume, float2* d_projft, float* d_projweights, int2 dimsproj, int rmax, float3* h_angles, int* h_ivolume, float3 magnification, float supersample, bool outputdecentered, uint batch);
extern "C" void __stdcall ProjectForward3DTex(uint64_t t_inputRe, uint64_t t_inputIm, float2* d_outputft, int3 dimsinput, int3 dimsoutput, float3* h_angles, float supersample, uint batch);
extern "C" void Scale(float* d_input, float* d_output, int3 dimsinput, int3 dimsoutput, uint batch, int planforw, int planback, float2* d_inputfft, float2* d_outputfft);
extern "C" void SphereMask(float* d_input, float* d_output, int3 dims, float radius, float sigma, bool decentered, uint batch);

//Projector.cpp
extern "C" void __stdcall InitProjector(int3 dims, int oversampling, float* h_data, float* h_initialized, int projdim);
extern "C" void __stdcall BackprojectorReconstructGPU(int3 dimsori, int3 dimspadded, int oversampling, float2* d_dataft, float* d_weights, char* c_symmetry, bool do_reconstruct_ctf, float* d_result, cufftHandle pre_planforw, cufftHandle pre_planback, cufftHandle pre_planforwctf, int griddingiterations);


//Functions.h
extern "C" __declspec(dllexport) void CreateTexture3DComplex(float2* d_data, int3 dims, uint64_t* h_textureid, uint64_t* h_arrayid, bool linearfiltering);
extern "C" __declspec(dllexport) void ProjectForwardTex(uint64_t t_inputRe, uint64_t t_inputIm, float2* d_outputft, int3 dimsinput, int2 dimsoutput, float3* h_angles, float supersample, uint batch);
extern "C" __declspec(dllexport) void DestroyTexture(uint64_t textureid, uint64_t arrayid);

//Wrappers

void ResizeMapGPU(MultidimArray<float> &img, int3 newDim);
void SphereMaskGPU(float* d_input, float* d_output, int3 dims, float radius, float sigma, bool decentered, uint batch);
void ResizeMapGPU(MultidimArray<float> &img, int2 newDim);
void Substract_GPU(MultidimArray<float> &img, float substrahend);
double SquaredSum(MultidimArray<float> &img);
void Substract_GPU(MultidimArray<float> &img, MultidimArray<RDOUBLE> &substrahend);
std::vector<float3> GetHealpixAnglesRad(int order, char * symmetry = "C1", float limittilt = -91);
void realspaceCTF(MultidimArray<RDOUBLE> &ctf, MultidimArray<RDOUBLE> &realCtf, int3 dims);
#endif
