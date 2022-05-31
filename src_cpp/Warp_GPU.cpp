#include "Warp_GPU.h"
#include <cassert>
#include "cudaHelpers.cuh"
#include "GTOM.cuh"

void ResizeMapGPU(MultidimArray<float> &img, int3 newDim)
{
	float * dout;
	cudaErrchk(cudaMalloc(&dout, Elements(newDim) * sizeof(float)));
	float * din;
	cudaErrchk(cudaMalloc(&din, img.nzyxdim*sizeof(float)));
	cudaErrchk(cudaMemcpy(din, img.data, img.nzyxdim * sizeof(float), cudaMemcpyHostToDevice));
	gtom::d_Scale(din, dout, gtom::toInt3(img.xdim, img.ydim, img.zdim), newDim, gtom::T_INTERP_FOURIER, NULL, NULL, 1);
	img.resizeNoCopy(newDim.z, newDim.y, newDim.x);
	cudaErrchk(cudaMemcpy(img.data, dout, img.nzyxdim * sizeof(float), cudaMemcpyDeviceToHost));
	cudaErrchk(cudaFree(din));
	cudaErrchk(cudaFree(dout));
}

void SphereMaskGPU(float * input, float * output, int3 dims, float radius, float sigma, bool decentered, uint batch)
{
	float * d_input = MallocDeviceFromHost(input, Elements(dims));
	float * d_output = MallocDevice(Elements(dims));
	SphereMask(d_input, d_output, dims, radius, sigma, decentered, batch);
	cudaMemcpy(output, d_output, Elements(dims)*sizeof(float),cudaMemcpyDeviceToHost);
	cudaFree(d_input);
	cudaFree(d_output);
}


void ResizeMapGPU(MultidimArray<float> &img, int2 newDim)
{
	float* dout = MallocDevice(newDim.x*newDim.y*img.zdim * sizeof(float));
	float* din = MallocDeviceFromHost(img.data, img.nzyxdim);

	Scale(din, dout, make_int3(img.xdim, img.ydim, 1), gtom::toInt3(newDim), img.zdim, 0, 0, NULL, NULL);
	img.resizeNoCopy(img.zdim, newDim.y, newDim.x);
	CopyDeviceToHost(dout, img.data, img.nzyxdim);
	FreeDevice(din);
	FreeDevice(dout);
}

double SquaredSum(MultidimArray<float> &img) {
	float* din = MallocDeviceFromHost(img.data, img.nzyxdim);
	float* dout;
	cudaErrchk(cudaMalloc(&dout, sizeof(float)*1))
	d_MultiplyByVector(din, din, din, img.nzyxdim, 1);
	
	d_SumMonolithic(din, dout, img.nzyxdim, 1);
	float output;
	cudaMemcpy(&output, dout, sizeof(float) * 1, cudaMemcpyDeviceToHost);
	cudaFree(din);
	cudaFree(dout);
	return output;
}

void Substract_GPU(MultidimArray<float> &img, MultidimArray<float> &substrahend) {
	assert(substrahend.nzyxdim == img.nzyxdim && "Shapes should match");
	float* din = MallocDeviceFromHost(img.data, img.nzyxdim);
	float* dSubtrahends = MallocDeviceFromHost(substrahend.data, substrahend.nzyxdim);

	SubtractFromSlices(din, dSubtrahends, din, img.nzyxdim, 1);

	cudaMemcpy(img.data, din, img.nzyxdim*sizeof(*din), cudaMemcpyDeviceToHost);
	cudaFree(din);
	cudaFree(dSubtrahends);

}


std::vector<float3> GetHealpixAnglesRad(int order, char * symmetry, float limittilt)
{
	int N = GetAnglesCount(order, symmetry, limittilt);
	
	float3 * c_array = (float3 *) malloc(sizeof(*c_array)*N);

	GetAngles(c_array, order, symmetry, limittilt);
	for (size_t i = 0; i < N; i++)
	{
		c_array[i] *= PI / 360;
	}
	std::vector<float3> output(c_array, c_array + N);
	return output;
}

void realspaceCTF(MultidimArray<RDOUBLE> &ctf, MultidimArray<RDOUBLE> &realCtf, int3 dims)
{
	realCtf.resizeNoCopy(ctf.zdim, dims.y, dims.x);
	idxtype batch = 1024;
	int3 sliceDim = { dims.x, dims.y, 1 };
	int ndims = DimensionCount(sliceDim);
	tfloat3 *shifts = (tfloat3 *)malloc(batch * sizeof(*shifts));
	for (size_t i = 0; i < batch; i++)
	{
		shifts[i] = { -dims.x / 2, -dims.y / 2, 0 };
	}
	tcomplex * h_ctf = (tcomplex *)malloc(ElementsFFT2(dims)*ctf.zdim * sizeof(tcomplex*));
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY3D(ctf) {
		h_ctf[k*YXSIZE(ctf) + (i * XSIZE(ctf)) + j] = { DIRECT_A3D_ELEM(ctf, k, i, j)*DIRECT_A3D_ELEM(ctf, k, i, j), 0.0f };
	}

	for (int idx = 0; idx < ctf.zdim;) {
		int batchElements = std::min(ctf.zdim - idx, batch);
		tcomplex * d_fft;
		cudaErrchk(cudaMalloc(&d_fft, ElementsFFT(sliceDim)*batchElements * sizeof(*d_fft)));
		cudaMemcpy(d_fft, h_ctf + idx * ElementsFFT(sliceDim), ElementsFFT(sliceDim)*batchElements * sizeof(*d_fft), cudaMemcpyHostToDevice);
		d_Shift(d_fft, d_fft, sliceDim, shifts, false, batchElements);
		float * d_result;
		cudaErrchk(cudaMalloc(&d_result, Elements(sliceDim)*batchElements * sizeof(*d_result)));
		d_IFFTC2R(d_fft, d_result, DimensionCount(sliceDim), sliceDim, batchElements);
		cudaErrchk(cudaMemcpy(realCtf.data + Elements(sliceDim) * idx, d_result, Elements(sliceDim)*batchElements * sizeof(float), cudaMemcpyDeviceToHost));

		cudaErrchk(cudaFree(d_fft));
		cudaErrchk(cudaFree(d_result));
		idx += batchElements;
	}
}