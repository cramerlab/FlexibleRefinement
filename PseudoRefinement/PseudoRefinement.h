#pragma once

#include "Types.h"
#include "PseudoProjector.h"



typedef  PseudoProjector* PseudoProjectorPTR;
typedef  float* floatPTR;


extern "C" __declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(int3 dims, float *atomCenters, float *atomWeights, float rAtoms, unsigned int nAtoms);

extern "C" __declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj, float* output, float* output_nrm, float3 angles, float shiftX, float shiftY, unsigned int batch);
extern "C" __declspec(dllexport) void __stdcall GetProjectionCTF(PseudoProjectorPTR proj, float* output, float* output_nrm, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border, float3 angles, float shiftX, float shiftY, unsigned int batch);

extern "C" __declspec(dllexport) float __stdcall DoARTStepMoved(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, float *atomPositions, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
extern "C" __declspec(dllexport) float __stdcall DoARTStepMovedCTF(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, float *atomPositions, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
extern "C" __declspec(dllexport) float __stdcall DoARTStepMovedCTF_DB(PseudoProjectorPTR proj, DOUBLE * Iexp, DOUBLE * Itheo, DOUBLE * Icorr, DOUBLE * Idiff, float3 * angles, float *atomPositions, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) float __stdcall DoARTStepCTF(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border,DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);


extern "C" __declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, float* outp);
extern "C" __declspec(dllexport) void __stdcall convolve(DOUBLE * img, DOUBLE * ctf, DOUBLE * outp, int3 dims);

extern "C" __declspec(dllexport) void __stdcall getGaussianTableFull(float * table, DOUBLE sigma, int interpoints);