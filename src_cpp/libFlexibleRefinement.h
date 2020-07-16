#pragma once

#include "Types.h"
#include "liblionImports.h"
#include "PseudoProjector.h"



typedef  PseudoProjector* PseudoProjectorPTR;
typedef  float* floatPTR;


extern "C" __declspec(dllexport) PseudoProjectorPTR __stdcall EntryPoint(int3 dims, RDOUBLE *atomCenters, RDOUBLE *atomWeights, RDOUBLE rAtoms, unsigned int nAtoms);

extern "C" __declspec(dllexport) void __stdcall GetProjection(PseudoProjectorPTR proj, RDOUBLE* output, RDOUBLE* output_nrm, float3 angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int batch);
extern "C" __declspec(dllexport) void __stdcall GetProjectionCTF(PseudoProjectorPTR proj, RDOUBLE* output, RDOUBLE* output_nrm, RDOUBLE * GaussTables, RDOUBLE * GaussTables2, RDOUBLE border, float3 angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int batch);

extern "C" __declspec(dllexport) float __stdcall DoARTStepMoved(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE *atomPositions, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
extern "C" __declspec(dllexport) float __stdcall DoARTStepMovedCTF(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE *atomPositions, RDOUBLE * GaussTables, RDOUBLE * GaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
extern "C" __declspec(dllexport) float __stdcall DoARTStepMovedCTF_DB(PseudoProjectorPTR proj, RDOUBLE * Iexp, RDOUBLE * Itheo, RDOUBLE * Icorr, RDOUBLE * Idiff, float3 * angles, RDOUBLE *atomPositions, RDOUBLE * GaussTables, RDOUBLE * GaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) float __stdcall DoARTStep(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);

extern "C" __declspec(dllexport) float __stdcall DoARTStepCTF(PseudoProjectorPTR proj, RDOUBLE * Iexp, float3 * angles, RDOUBLE * GaussTables, RDOUBLE * GaussTables2, RDOUBLE border,RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);


extern "C" __declspec(dllexport) void __stdcall GetIntensities(PseudoProjectorPTR proj, RDOUBLE* outp);
extern "C" __declspec(dllexport) void __stdcall convolve(RDOUBLE * img, RDOUBLE * ctf, RDOUBLE * outp, int3 dims);

extern "C" __declspec(dllexport) void __stdcall getGaussianTableFull(RDOUBLE * table, RDOUBLE sigma, int interpoints);