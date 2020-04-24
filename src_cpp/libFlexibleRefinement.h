#pragma once

#include "Types.h"
#include "liblionImports.h"
#include "PseudoProjector.h"



typedef  PseudoProjector* PseudoProjectorPTR;
typedef  float* floatPTR;


extern "C" __declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(int3 dims, DOUBLE *atomCenters, DOUBLE *atomWeights, DOUBLE rAtoms, unsigned int nAtoms);

extern "C" __declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj, DOUBLE* output, DOUBLE* output_nrm, float3 angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int batch);
extern "C" __declspec(dllexport) void __stdcall GetProjectionCTF(PseudoProjectorPTR proj, DOUBLE* output, DOUBLE* output_nrm, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border, float3 angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int batch);

extern "C" __declspec(dllexport) float __stdcall DoARTStepMoved(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE *atomPositions, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
extern "C" __declspec(dllexport) float __stdcall DoARTStepMovedCTF(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE *atomPositions, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
extern "C" __declspec(dllexport) float __stdcall DoARTStepMovedCTF_DB(PseudoProjectorPTR proj, DOUBLE * Iexp, DOUBLE * Itheo, DOUBLE * Icorr, DOUBLE * Idiff, float3 * angles, DOUBLE *atomPositions, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) float __stdcall DoARTStepCTF(PseudoProjectorPTR proj, DOUBLE * Iexp, float3 * angles, DOUBLE * GaussTables, DOUBLE * GaussTables2, DOUBLE border,DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);


extern "C" __declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, DOUBLE* outp);
extern "C" __declspec(dllexport) void __stdcall convolve(DOUBLE * img, DOUBLE * ctf, DOUBLE * outp, int3 dims);

extern "C" __declspec(dllexport) void __stdcall getGaussianTableFull(DOUBLE * table, DOUBLE sigma, int interpoints);