#include "gpu_project.cuh"
//#include "device_atomic_functions.hpp"
using namespace gtom;
#define DEG2RAD(d) ((d) * PI / 180)
__global__ void RealspacePseudoProjectForwardKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, int2 dimsproj, glm::mat3* d_rotations);

void RealspacePseudoProjectForward(float3* d_atomPositions,
	float *d_atomIntensities,
	unsigned int nAtoms,
	 int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
	float supersample,
	float3* h_angles,
	int batch)
{
	d_ValueFill(d_projections, Elements2(dimsproj) * batch, 0.0f);

	glm::mat3* d_matrices;

	glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
	for (int i = 0; i < batch; i++)
		h_matrices[i] = Matrix3Euler(h_angles[i]);
	d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
	free(h_matrices);

	dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), batch, 1);
	uint elements = 128;

	RealspacePseudoProjectForwardKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, nAtoms, dimsvolume, supersample, d_projections, dimsproj, d_matrices);

	cudaFree(d_matrices);
}

__global__ void RealspacePseudoProjectForwardKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, int2 dimsproj, glm::mat3* d_rotations)
{
	d_projections += Elements2(dimsproj) * blockIdx.y;


	uint slice = nAtoms;

	glm::mat3 rotation = d_rotations[blockIdx.y];
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);

	for (uint id = blockIdx.x * blockDim.x + threadIdx.x; id < Elements(dimsvolume); id += gridDim.x * blockDim.x)
	{
		uint idxAtom = id % slice;

		uint idxProjection = id / slice;

		float weight = d_atomIntensities[idxAtom];

		glm::vec3 pos = glm::vec3(d_atomPositions[idxAtom].x, d_atomPositions[idxAtom].y, d_atomPositions[idxAtom].z)*supersample;
		pos -= volumecenter;
		pos = rotation * pos;
		pos += projectioncenter;
		pos*supersample;
		// Bilinear interpolation
		int X0 = floor(pos.x);
		float ix = pos.x - X0;
		int X1 = X0 + 1;

		int Y0 = floor(pos.y);
		float iy = pos.y - Y0;
		int Y1 = Y0 + 1;

		float v0 = 1.0f - iy;
		float v1 = iy;

		float v00 = (1.0f - ix) * v0;
		float v10 = ix * v0;
		float v01 = (1.0f - ix) * v1;
		float v11 = ix * v1;

		atomicAdd((d_projections + (Y0*dimsproj.x + X0)), weight * v00);
		atomicAdd((d_projections + (Y0*dimsproj.x + X1)), weight * v01);
		atomicAdd((d_projections + (Y1*dimsproj.x + X0)), weight * v10);
		atomicAdd((d_projections + (Y1*dimsproj.x + X1)), weight * v11);
	}
}