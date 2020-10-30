#include "cudaHelpers.cuh"
#include "Generics.cuh"
void cudaAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
void cufftAssert(cufftResult_t code, const char *file, int line, bool abort)
{
	if (code != CUFFT_SUCCESS)
	{
		fprintf(stderr, "CUFFTassert: %s %s %d\n", _cudaGetErrorEnum(code), file, line);
		if (abort) exit(code);
	}
}


__global__ void OwnComplexDivideByVectorKernel(tcomplex* d_input, tfloat* d_divisors, tcomplex* d_output, size_t elements);

__global__ void OwnComplexDivideSafeByVectorKernel(tcomplex* d_input, tfloat* d_divisors, tcomplex* d_output, size_t elements)
{
	d_input += elements * blockIdx.y;
	d_output += elements * blockIdx.y;

	tfloat val;
	for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
		id < elements;
		id += blockDim.x * gridDim.x)
	{
		val = d_divisors[id];
		if (abs(val) < 1e-15)
			val = 0;
		else
			val = (tfloat)1 / val;
		d_output[id] = d_input[id] * val;
	}
}

void d_OwnComplexDivideSafeByVector(tcomplex* d_input, tfloat* d_divisors, tcomplex* d_output, size_t elements, int batch)
{
	size_t TpB = tmin((size_t)256, elements);
	dim3 grid = dim3(tmin((elements + TpB - 1) / TpB, (size_t)32768), batch);
	OwnComplexDivideSafeByVectorKernel << <grid, (uint)TpB >> > (d_input, d_divisors, d_output, elements);
}

#ifdef _CUFFT_H_
// cuFFT API errors
const char *_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}


#endif
__device__ float d_Lerp(float a, float b, float x)
{
	return a + (b - a) * x;
}

cufftHandle d_OwnIFFTC2RGetPlan(int const ndimensions, int3 const dimensions, int batch);

void d_OwnIFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan, int3 dimensions, int batch)
{
#ifdef GTOM_DOUBLE
	cufftExecZ2D(*plan, d_input, d_output);
#else
	cufftErrchk(cufftExecC2R(*plan, d_input, d_output));
#endif

	gtom::d_MultiplyByScalar(d_output, d_output, Elements(dimensions) * batch, 1.0f / (float)Elements(dimensions));
}

void d_OwnIFFTC2R(tcomplex* const d_input, tfloat* const d_output, cufftHandle* plan)
{
#ifdef GTOM_DOUBLE
	cufftExecZ2D(*plan, d_input, d_output);
#else
	cufftErrchk(cufftExecC2R(*plan, d_input, d_output));
#endif
}

void d_OwnIFFTC2R(tcomplex* const d_input, tfloat* const d_output, int const ndimensions, int3 const dimensions, int batch, bool renormalize)
{
	cufftHandle plan = d_OwnIFFTC2RGetPlan(ndimensions, dimensions, batch);
	if (renormalize)
		d_OwnIFFTC2R(d_input, d_output, &plan, dimensions, batch);
	else
		d_OwnIFFTC2R(d_input, d_output, &plan);
	cufftErrchk(cufftDestroy(plan));
}

cufftHandle d_OwnIFFTC2RGetPlan(int const ndimensions, int3 const dimensions, int batch)
{
	cufftHandle plan;
	cufftType direction = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	int n[3] = { dimensions.z, dimensions.y, dimensions.x };

	cufftErrchk(cufftPlanMany(&plan, ndimensions, n + (3 - ndimensions),
		NULL, 1, 0,
		NULL, 1, 0,
		direction, batch));

	//cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE);

	return plan;
}
__global__ void MaxOpKernel(float* d_input1, float d_input2, float* d_output, size_t elements);
__global__ void SphereMaskFTKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, int radius2);

void d_SphereMaskFT(tcomplex* d_input, tcomplex* d_output, int3 dims, int radius, uint batch)
{
	int TpB = tmin(128, NextMultipleOf(dims.x, 32));
	dim3 grid = dim3(dims.y, dims.z, batch);
	SphereMaskFTKernel << <grid, TpB >> > (d_input, d_output, dims, radius * radius);
}

__global__ void SphereMaskFTKernel(tcomplex* d_input, tcomplex* d_output, int3 dims, int radius2)
{
	int z = blockIdx.y;
	int y = blockIdx.x;

	d_input += blockIdx.z * ElementsFFT(dims) + (z * dims.y + y) * (dims.x / 2 + 1);
	d_output += blockIdx.z * ElementsFFT(dims) + (z * dims.y + y) * (dims.x / 2 + 1);

	int zp = z < dims.z / 2 + 1 ? z : z - dims.x;
	zp *= zp;
	int yp = y < dims.y / 2 + 1 ? y : y - dims.x;
	yp *= yp;

	for (int x = threadIdx.x; x < dims.x / 2 + 1; x += blockDim.x)
	{
		int r = x * x + yp + zp;

		if (r < radius2)
			d_output[x] = d_input[x];
		else
			d_output[x] = { 0, 0 };
	}
}

void d_MaxOp(float* d_input1, float input2, float* d_output, size_t elements)
{
	size_t TpB = tmin((size_t)256, elements);
	size_t totalblocks = tmin((elements + TpB - 1) / TpB, (size_t)8192);
	dim3 grid = dim3((uint)totalblocks);
	MaxOpKernel << <grid, (uint)TpB >> > (d_input1, input2, d_output, elements);
}

 __global__ void MaxOpKernel(float* d_input1, float d_input2, float* d_output, size_t elements)
{
	for (size_t id = blockIdx.x * blockDim.x + threadIdx.x;
		id < elements;
		id += blockDim.x * gridDim.x)
		d_output[id] = fmaxf(d_input1[id], d_input2);
}
