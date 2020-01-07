#pragma once

#include "Types.h"
#include "PseudoProjector.h"



typedef  PseudoProjector* PseudoProjectorPTR;
typedef  float* floatPTR;


extern "C" __declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(float* projections, float* angles, int3 dims, float *atomCenters, float *atomWeights, float rAtoms, unsigned int nAtoms);

extern "C" __declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj, float* output, float* output_nrm, float3 angles, float shiftX, float shiftY, unsigned int batch);

extern "C" __declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, float* outp);