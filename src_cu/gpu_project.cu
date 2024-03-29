#include "gpu_project.cuh"
#include "cudaHelpers.cuh"
using namespace gtom;

__global__ void RealspacePseudoProjectForwardKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, int2 dimsproj, glm::mat3* d_rotations);
__global__ void RealspacePseudoProjectForwardKernel(float3* d_atomPositions, float* d_atomIntensities, int *d_positionMapping, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, int2 dimsproj, glm::mat3* d_rotations);
__global__ void RealspacePseudoProjectBackwardKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, unsigned int nProjections, int2 dimsproj, glm::mat3* d_rotations);
__global__ void RealspacePseudoProjectBackwardKernel(float3* d_atomPositions, float* d_atomIntensities, int *d_positionMapping, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, unsigned int nProjections, int2 dimsproj, glm::mat3* d_rotations);
__global__ void RealspaceVolumeUpdateKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, float* d_superVolumeUpdates, int3 dimsvolume, float supersample);
__global__ void RealspacePseudoCreateMaskKernel(float3* d_atomPositions, float* d_atomIntensities, int *d_positionMapping, unsigned int nAtoms, int3 dimsvolume, float* d_projections, int2 dimsproj, glm::mat3* d_rotations);


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

	//Only parallelise for the different projections
	dim3 grid = dim3(1, batch, 1);
	uint elements = 1;

	// This layout had a parallelisation over the atomnumber as well, but required atomic add operation
	/*dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), batch, 1);
	uint elements = 128;*/
	

	RealspacePseudoProjectForwardKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, nAtoms, dimsvolume, supersample, d_projections, dimsproj, d_matrices);

	cudaFree(d_matrices);
}

__global__ void RealspacePseudoProjectForwardKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, int2 dimsproj, glm::mat3* d_rotations)
{
	uint idxProjection = blockIdx.y;

	d_projections += Elements2(dimsproj) * idxProjection;


	uint slice = nAtoms;

	glm::mat3 rotation = d_rotations[idxProjection];
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);

	for (uint id = blockIdx.x; id < nAtoms; id += blockDim.x)
	{
		uint idxAtom = id ;

		float weight = d_atomIntensities[idxAtom];
		if (weight == 0.0f)
			continue;
		float3 atomPos = d_atomPositions[idxAtom];
		glm::vec3 pos = glm::vec3(atomPos.x, atomPos.y, atomPos.z)*supersample;
		pos -= volumecenter;
		pos = rotation * pos;
		pos += volumecenter;

		// Bilinear interpolation
		int X0 = floor(pos.x);
		float ix = pos.x - X0;
		int X1 = X0 + 1;

		int Y0 = floor(pos.y);
		float iy = pos.y - Y0;
		int Y1 = Y0 + 1;

		float v0 = 1.0f - ix;
		float v1 = ix;

		float v00 = (1.0f - iy) * v0;
		float v10 = iy * v0;
		float v01 = (1.0f - iy) * v1;
		float v11 = iy * v1;

		d_projections[Y0*dimsproj.x + X0] += weight * v00;
		d_projections[Y0*dimsproj.x + X1] += weight * v01;
		d_projections[Y1*dimsproj.x + X0] += weight * v10;
		d_projections[Y1*dimsproj.x + X1] += weight * v11;
	}
}

void RealspacePseudoProjectBackward(float3* d_atomPositions,
	float *d_atomIntensities,
	unsigned int nAtoms,
	int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
	float supersample,
	float3* h_angles,
	int batch)
{


	glm::mat3* d_matrices;

	glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
	for (int i = 0; i < batch; i++)
		h_matrices[i] = Matrix3Euler(h_angles[i]);
	d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
	free(h_matrices);



	//To be most efficient (no atomic add operations), we can parallelize over atoms and have each thread process all projections for one atom.
	dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), 1, 1);
	uint elements = 128;

	RealspacePseudoProjectBackwardKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, nAtoms, dimsvolume, supersample, d_projections, batch, dimsproj, d_matrices);
	cudaFree(d_matrices);


}

__global__ void RealspacePseudoProjectBackwardKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, unsigned int nProjections, int2 dimsproj, glm::mat3* d_rotations)
{
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);
	for (unsigned int idxAtom = blockIdx.x*blockDim.x + threadIdx.x; idxAtom < nAtoms; idxAtom += blockDim.x*gridDim.x) {
		double update = 0.0;
		for (size_t idxProjection = 0; idxProjection < nProjections; idxProjection++)
		{
			float* d_projectionSlice = d_projections + idxProjection * Elements2(dimsproj);
			glm::mat3 rotation = d_rotations[idxProjection];
			float3 atomPos = d_atomPositions[idxAtom];
			glm::vec3 pos = glm::vec3(atomPos.x, atomPos.y, atomPos.z)*supersample;
			pos -= volumecenter;
			pos = rotation * pos;
			pos += volumecenter;

			int X0 = (int)pos.x;
			float ix = pos.x - X0;
			int X1 = X0 + 1;

			int Y0 = (int)pos.y;
			float iy = pos.y - Y0;
			int Y1 = Y0 + 1;

			double v00 = d_projectionSlice[Y0 * dimsproj.x + X0];
			double v01 = d_projectionSlice[Y0 * dimsproj.x + X1];
			double v10 = d_projectionSlice[Y1 * dimsproj.x + X0];
			double v11 = d_projectionSlice[Y1 * dimsproj.x + X1];


			double v0 = d_Lerp(v00, v01, ix);
			double v1 = d_Lerp(v10, v11, ix);

			double v = d_Lerp(v0, v1, iy);
			update += v;		

		}
		d_atomIntensities[idxAtom] += update;
	}
}



void RealspaceVolumeUpdate(float3* d_atomPositions,
	float *d_atomIntensities,
	unsigned int nAtoms,
	float* d_superVolumeUpdates,
	int3 dimsvolume,
	float supersample)
{

	//To be most efficient (no atomic add operations), we can parallelize over atoms and have each thread process all projections for one atom.
	dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), 1, 1);
	uint elements = 128;

	RealspaceVolumeUpdateKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, nAtoms, d_superVolumeUpdates, dimsvolume, supersample);
}

__global__ void RealspaceVolumeUpdateKernel(float3* d_atomPositions, float* d_atomIntensities, unsigned int nAtoms, float* d_superVolumeUpdates, int3 dimsvolume, float supersample) {

	for (unsigned int idxAtom = blockIdx.x*blockDim.x + threadIdx.x; idxAtom < nAtoms; idxAtom += blockDim.x*gridDim.x) {
		double update = 0.0;
		float3 atomPos = d_atomPositions[idxAtom];
		glm::vec3 pos = glm::vec3(atomPos.x, atomPos.y, atomPos.z)*supersample;

		int3 dimsSuper = dimsvolume * supersample;

		size_t xydims = dimsSuper.x*dimsSuper.y;

		int X0 = (int)pos.x;
		float ix = pos.x - X0;
		int X1 = X0 + 1;

		int Y0 = (int)pos.y;
		float iy = pos.y - Y0;
		int Y1 = Y0 + 1;

		int Z0 = (int)pos.z;
		float iz = pos.z - Z0;
		int Z1 = Z0 + 1;

		double v000 = d_superVolumeUpdates[Z0*xydims + Y0 * dimsSuper.x + X0];
		double v100 = d_superVolumeUpdates[Z1*xydims + Y0 * dimsSuper.x + X0];
		double v001 = d_superVolumeUpdates[Z0*xydims + Y0 * dimsSuper.x + X1];
		double v101 = d_superVolumeUpdates[Z1*xydims + Y0 * dimsSuper.x + X1];
		double v010 = d_superVolumeUpdates[Z0*xydims + Y1 * dimsSuper.x + X0];
		double v110 = d_superVolumeUpdates[Z1*xydims + Y1 * dimsSuper.x + X0];
		double v011 = d_superVolumeUpdates[Z0*xydims + Y1 * dimsSuper.x + X1];
		double v111 = d_superVolumeUpdates[Z1*xydims + Y1 * dimsSuper.x + X1];

		double v00 = d_Lerp(v000, v100, iz);
		double v01 = d_Lerp(v001, v101, iz);
		double v10 = d_Lerp(v010, v110, iz);
		double v11 = d_Lerp(v011, v111, iz);


		double v0 = d_Lerp(v00, v01, ix);
		double v1 = d_Lerp(v10, v11, ix);

		double v = d_Lerp(v0, v1, iy);
		update += v;


		d_atomIntensities[idxAtom] += update;
	}
}

/* 
Functions with mapping
*/

void RealspacePseudoProjectForward(float3* d_atomPositions,
	float *d_atomIntensities,
	int *d_positionMapping,
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

	//Only parallelise for the different projections
	dim3 grid = dim3(1, batch, 1);
	uint elements = 1;

	// This layout had a parallelisation over the atomnumber as well, but required atomic add operation
	/*dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), batch, 1);
	uint elements = 128;*/


	RealspacePseudoProjectForwardKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, d_positionMapping, nAtoms, dimsvolume, supersample, d_projections, dimsproj, d_matrices);

	cudaFree(d_matrices);
}

__global__ void RealspacePseudoProjectForwardKernel(float3* d_atomPositions, float* d_atomIntensities, int *d_positionMapping, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, int2 dimsproj, glm::mat3* d_rotations)
{
	uint idxProjection = blockIdx.y;

	d_projections += Elements2(dimsproj) * idxProjection;


	uint slice = nAtoms;

	glm::mat3 rotation = d_rotations[idxProjection];
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);

	for (uint id = blockIdx.x; id < nAtoms; id += blockDim.x)
	{
		uint idxAtom = id;

		float weight = d_atomIntensities[idxAtom];
		if (weight == 0.0f)
			continue;
		int correctIDX = d_positionMapping[idxProjection] * nAtoms + idxAtom;

		float3 atomPos = d_atomPositions[correctIDX];
		glm::vec3 pos = glm::vec3(atomPos.x, atomPos.y, atomPos.z)*supersample;
		pos -= volumecenter;
		pos = rotation * pos;
		pos += volumecenter;

		// Bilinear interpolation
		int X0 = floor(pos.x);
		float ix = pos.x - X0;
		int X1 = X0 + 1;

		int Y0 = floor(pos.y);
		float iy = pos.y - Y0;
		int Y1 = Y0 + 1;

		float v0 = 1.0f - ix;
		float v1 = ix;

		float v00 = (1.0f - iy) * v0;
		float v10 = iy * v0;
		float v01 = (1.0f - iy) * v1;
		float v11 = iy * v1;

		d_projections[Y0*dimsproj.x + X0] += weight * v00;
		d_projections[Y0*dimsproj.x + X1] += weight * v01;
		d_projections[Y1*dimsproj.x + X0] += weight * v10;
		d_projections[Y1*dimsproj.x + X1] += weight * v11;
	}
}


void RealspacePseudoProjectBackward(float3* d_atomPositions,
	float *d_atomIntensities,
	int *d_positionMapping,
	unsigned int nAtoms,
	int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
	float supersample,
	float3* h_angles,
	int batch)
{


	glm::mat3* d_matrices;

	glm::mat3* h_matrices = (glm::mat3*)malloc(sizeof(glm::mat3) * batch);
	for (int i = 0; i < batch; i++)
		h_matrices[i] = Matrix3Euler(h_angles[i]);
	d_matrices = (glm::mat3*)CudaMallocFromHostArray(h_matrices, sizeof(glm::mat3) * batch);
	free(h_matrices);



	//To be most efficient (no atomic add operations), we can parallelize over atoms and have each thread process all projections for one atom.
	dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), 1, 1);
	uint elements = 128;

	RealspacePseudoProjectBackwardKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, d_positionMapping, nAtoms, dimsvolume, supersample, d_projections, batch, dimsproj, d_matrices);
	cudaFree(d_matrices);


}

__global__ void RealspacePseudoProjectBackwardKernel(float3* d_atomPositions, float* d_atomIntensities, int *d_positionMapping, unsigned int nAtoms, int3 dimsvolume, float supersample, float* d_projections, unsigned int nProjections, int2 dimsproj, glm::mat3* d_rotations)
{
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);
	for (unsigned int idxAtom = blockIdx.x*blockDim.x + threadIdx.x; idxAtom < nAtoms; idxAtom += blockDim.x*gridDim.x) {
		double update = 0.0;
		for (size_t idxProjection = 0; idxProjection < nProjections; idxProjection++)
		{
			float* d_projectionSlice = d_projections + idxProjection * Elements2(dimsproj);
			glm::mat3 rotation = d_rotations[idxProjection];
			float3 atomPos = d_atomPositions[d_positionMapping[idxProjection] * nAtoms + idxAtom];
			glm::vec3 pos = glm::vec3(atomPos.x, atomPos.y, atomPos.z)*supersample;
			pos -= volumecenter;
			pos = rotation * pos;
			pos += volumecenter;

			int X0 = (int)pos.x;
			float ix = pos.x - X0;
			int X1 = X0 + 1;

			int Y0 = (int)pos.y;
			float iy = pos.y - Y0;
			int Y1 = Y0 + 1;

			double v00 = d_projectionSlice[Y0 * dimsproj.x + X0];
			double v01 = d_projectionSlice[Y0 * dimsproj.x + X1];
			double v10 = d_projectionSlice[Y1 * dimsproj.x + X0];
			double v11 = d_projectionSlice[Y1 * dimsproj.x + X1];


			double v0 = d_Lerp(v00, v01, ix);
			double v1 = d_Lerp(v10, v11, ix);

			double v = d_Lerp(v0, v1, iy);
			update += v;

		}
		d_atomIntensities[idxAtom] += update;
	}
}



void RealspacePseudoCreateMask(float3* d_atomPositions,
	float *d_atomIntensities,
	int *d_positionMapping,
	unsigned int nAtoms,
	int3 dimsvolume,
	float* d_projections,
	int2 dimsproj,
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

	//Only parallelise for the different projections
	dim3 grid = dim3(1, batch, 1);
	uint elements = 1;

	// This layout had a parallelisation over the atomnumber as well, but required atomic add operation
	/*dim3 grid = dim3(tmin(1024, (nAtoms + 127) / 128), batch, 1);
	uint elements = 128;*/


	RealspacePseudoCreateMaskKernel << <grid, elements >> > (d_atomPositions, d_atomIntensities, d_positionMapping, nAtoms, dimsvolume, d_projections, dimsproj, d_matrices);

	cudaFree(d_matrices);
}

__global__ void RealspacePseudoCreateMaskKernel(float3* d_atomPositions, float* d_atomIntensities, int *d_positionMapping, unsigned int nAtoms, int3 dimsvolume, float* d_projections, int2 dimsproj, glm::mat3* d_rotations)
{
	uint idxProjection = blockIdx.y;

	d_projections += Elements2(dimsproj) * idxProjection;


	uint slice = nAtoms;

	glm::mat3 rotation = d_rotations[idxProjection];
	glm::vec3 volumecenter = glm::vec3(dimsvolume.x / 2, dimsvolume.y / 2, dimsvolume.z / 2);
	glm::vec3 projectioncenter = glm::vec3(dimsproj.x / 2, dimsproj.y / 2, 0);

	for (uint id = blockIdx.x; id < nAtoms; id += blockDim.x)
	{
		uint idxAtom = id;

		float weight = d_atomIntensities[idxAtom];
		if (weight == 0.0f)
			continue;
		int correctIDX = d_positionMapping[idxProjection] * nAtoms + idxAtom;

		float3 atomPos = d_atomPositions[correctIDX];
		glm::vec3 pos = glm::vec3(atomPos.x, atomPos.y, atomPos.z);
		pos -= volumecenter;
		pos = rotation * pos;
		pos += volumecenter;

		// Bilinear interpolation
		int X0 = floor(pos.x);
		int X1 = X0 + 1;
		int Y0 = floor(pos.y);
		int Y1 = Y0 + 1;


		d_projections[Y0*dimsproj.x + X0] = 1;
		d_projections[Y0*dimsproj.x + X1] = 1;
		d_projections[Y1*dimsproj.x + X0] = 1;
		d_projections[Y1*dimsproj.x + X1] = 1;
	}
}

