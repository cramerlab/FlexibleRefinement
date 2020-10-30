#pragma once
#include "liblionImports.h"
#include "Prerequisites.cuh"
#include "Warp_GPU.h"
#include "cudaHelpers.cuh"
class Projector
{

public:
	int3 Dims, DimsOversampled;
	int Oversampling;

	float2 * d_data;
	float * d_weights;

	unsigned long t_DataRe, t_DataIm;
	unsigned long a_DataRe, a_DataIm;

	Projector(int3 dims, int oversampling)
	{
		Dims = dims;
		Oversampling = oversampling;

		int Oversampled = 2 * (oversampling * (Dims.x / 2) + 1) + 1;
		DimsOversampled = { Oversampled, Oversampled, Oversampled };

		cudaErrchk(cudaMalloc(&d_data, sizeof(*d_data)*Elements(DimsOversampled)));
		cudaErrchk(cudaMalloc(&d_weights, sizeof(*d_data)*Elements(DimsOversampled)));
	}



	void BackProject(float2 * d_projft, float * d_projweights, int3 projDim, float3 * angles, idxtype numAngles, float3 magnification)
	{
		gtom::d_rlnBackproject(d_data,
			d_weights,
			DimsOversampled,
			d_projft,
			d_projweights,
			projDim,
			projDim.x / 2,
			(tfloat3*)angles,
			NULL,
			magnification,
			Oversampling,
			false,
			(uint)numAngles);
	}

	float * d_Reconstruct(bool isctf, char * symmetry = "C1", int planForw = -1, int planBack = -1, int planForwCTF = -1, int griddingiterations = 10)
	{
		float * d_reconstruction;
		if (isctf) {
			cudaErrchk(cudaMalloc(&d_reconstruction, ElementsFFT(Dims) * sizeof(*d_reconstruction)));
		}
		else {
			cudaErrchk(cudaMalloc(&d_reconstruction, Elements(Dims) * sizeof(*d_reconstruction)));
		}
		BackprojectorReconstructGPU(Dims,
			DimsOversampled,
			Oversampling,
			d_data,
			d_weights,
			symmetry,
			isctf,
			d_reconstruction,
			planForw,
			planBack,
			planForwCTF,
			griddingiterations);

		return d_reconstruction;
	}

	void d_Reconstruct(float * d_reconstruction, char * symmetry = "C1", int planForw = -1, int planBack = -1, int planForwCTF = -1, int griddingiterations = 10)
	{
		BackprojectorReconstructGPU(Dims,
			DimsOversampled,
			Oversampling,
			d_data,
			d_weights,
			symmetry,
			false,
			d_reconstruction,
			planForw,
			planBack,
			planForwCTF,
			griddingiterations);
	}

};
