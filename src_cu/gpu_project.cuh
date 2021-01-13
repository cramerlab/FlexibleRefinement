#ifndef GPU_PROJECT
#define GPU_PROJECT
#include "GTOM.cuh"
#include "cudaHelpers.cuh"
using namespace gtom;



void RealspacePseudoProjectForward(float3* d_atomPositions,
	float *d_atomIntensities,
	unsigned int nAtoms,
	int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
	float supersample,
	float3* h_angles,
	int batch);

void RealspacePseudoProjectForward(float3* d_atomPositions,
	float *d_atomIntensities,
	int *d_positionMapping,
	unsigned int nAtoms,
	int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
	float supersample,
	float3* h_angles,
	int batch);

void RealspacePseudoProjectBackward(float3* d_atomPositions,
	float *d_atomIntensities,
	int *d_positionMapping,
	unsigned int nAtoms,
	int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
	float supersample,
	float3* h_angles,
	int batch);

void RealspacePseudoProjectBackward(float3* d_atomPositions,	float *d_atomIntensities, unsigned int nAtoms, int3 dimsvolume,	float* d_projections, int2 dimsproj, float supersample, float3* h_angles, int batch);

void RealspaceVolumeUpdate(float3* d_atomPositions, float *d_atomIntensities, unsigned int nAtoms, float* d_superVolumeUpdates, int3 dimsvolume, float supersample);

#endif // !GPU_PROJECT