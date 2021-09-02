
#include "PseudoProjector.h"
#include "gpu_project.cuh"
#include "cudaHelpers.cuh"


static inline void outputDeviceAsImage(tcomplex *d_data, int3 Dims, FileName outName) {

	float *d_abs;
	cudaErrchk(cudaMalloc(&d_abs, ElementsFFT(Dims) * sizeof(*d_abs)));
	d_Abs(d_data, d_abs, ElementsFFT(Dims));

	MultidimArray<float> h_data(Dims.z, Dims.y, ElementsFFT1(Dims.x));
	cudaErrchk(cudaMemcpy(h_data.data, d_abs, ElementsFFT(Dims) * sizeof(*d_abs), cudaMemcpyDeviceToHost));
	MRCImage<float> h_im(h_data);
	h_im.writeAs<float>(outName, true);
	cudaErrchk(cudaFree(d_abs));
}

static inline void outputDeviceAsImage(float *d_data, int3 Dims, FileName outName, bool isFT = false) {
	MultidimArray<float> h_data(Dims.z, Dims.y, isFT ? (Dims.x / 2 + 1) : Dims.x);
	cudaErrchk(cudaMemcpy(h_data.data, d_data, isFT ? ElementsFFT(Dims) : Elements(Dims) * sizeof(*d_data), cudaMemcpyDeviceToHost));
	MRCImage<float> h_im(h_data);
	h_im.writeAs<float>(outName, true);
}

static inline void outputAsImage(MultidimArray<float> h_data, FileName outName) {
	MRCImage<float> h_im(h_data);
	h_im.writeAs<float>(outName, true);
}

RDOUBLE PseudoProjector::ART_single_image(const MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &Itheo, MultidimArray<RDOUBLE> &Icorr, MultidimArray<RDOUBLE> &Idiff, RDOUBLE rot, RDOUBLE tilt, RDOUBLE psi, RDOUBLE shiftX, RDOUBLE shiftY)
{
	Idiff.initZeros();
	Itheo.initZeros();
	Icorr.initZeros();
	Matrix2D<RDOUBLE> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_FORWARD);
	//Idiff.resize(Iexp);

	RDOUBLE mean_error = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
	{
		// Compute difference image and error
		auto Iexp_i_j = DIRECT_A2D_ELEM(Iexp, i, j);
		auto Itheo_i_j = DIRECT_A2D_ELEM(Itheo, i, j);

		DIRECT_A2D_ELEM(Idiff, i, j) = DIRECT_A2D_ELEM(Iexp, i, j) - DIRECT_A2D_ELEM(Itheo, i, j);
		mean_error += DIRECT_A2D_ELEM(Idiff, i, j) * DIRECT_A2D_ELEM(Idiff, i, j);

		// Compute the correction image
		auto Icorr_i_j_a = DIRECT_A2D_ELEM(Icorr, i, j);
		DIRECT_A2D_ELEM(Icorr, i, j) = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);

		auto Icorr_i_j_b = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
		auto Icorr_i_j_c = this->lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
		DIRECT_A2D_ELEM(Icorr, i, j) =
			this->lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
	}
	mean_error /= YXSIZE(Iexp);

	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_BACKWARD);
	return mean_error;
}

RDOUBLE PseudoProjector::ART_single_image(const MultidimArray<RDOUBLE>& Iexp, RDOUBLE rot, RDOUBLE tilt, RDOUBLE psi, RDOUBLE shiftX, RDOUBLE shiftY)
{
	MultidimArray<RDOUBLE> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Matrix2D<RDOUBLE> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_FORWARD);
	Idiff.resize(Iexp);

	RDOUBLE mean_error = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
	{
		// Compute difference image and error

		DIRECT_A2D_ELEM(Idiff, i, j) = DIRECT_A2D_ELEM(Iexp, i, j) - DIRECT_A2D_ELEM(Itheo, i, j);
		mean_error += DIRECT_A2D_ELEM(Idiff, i, j) * DIRECT_A2D_ELEM(Idiff, i, j);
		// Compute the correction image
		DIRECT_A2D_ELEM(Icorr, i, j) = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
		DIRECT_A2D_ELEM(Icorr, i, j) =
			this->lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
	}
	mean_error /= YXSIZE(Iexp);

	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_BACKWARD);
	return mean_error;
}

RDOUBLE PseudoProjector::ART_batched(const MultidimArray<RDOUBLE> &Iexp, idxtype batchSize, float3 *angles, RDOUBLE shiftX, RDOUBLE shiftY)
{
	MultidimArray<RDOUBLE> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Idiff.resize(Iexp, false);
	Matrix2D<RDOUBLE> *EulerVec = new Matrix2D<RDOUBLE>[batchSize];
	RDOUBLE mean_error = 0;
#pragma omp parallel
	{
		//Initialize array for the slice view
		MultidimArray<RDOUBLE> tmpItheo, tmpIcorr;
		tmpItheo.xdim = tmpIcorr.xdim = Iexp.xdim;
		tmpItheo.ydim = tmpIcorr.ydim = Iexp.xdim;
		tmpItheo.destroyData = tmpIcorr.destroyData = false;
		tmpItheo.yxdim = tmpItheo.nzyxdim = tmpItheo.zyxdim = tmpItheo.xdim*tmpItheo.ydim;
		tmpIcorr.yxdim = tmpIcorr.nzyxdim = tmpIcorr.zyxdim = tmpIcorr.xdim*tmpItheo.ydim;
#pragma omp for
		for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
			tmpItheo.data = &(A3D_ELEM(Itheo, batchIdx, 0, 0));
			tmpIcorr.data = &(A3D_ELEM(Icorr, batchIdx, 0, 0));
			Euler_angles2matrix(angles[batchIdx].x, angles[batchIdx].y, angles[batchIdx].z, EulerVec[batchIdx]);
			this->project_Pseudo(tmpItheo, tmpIcorr,
				EulerVec[batchIdx], shiftX, shiftY, PSEUDO_FORWARD);



			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
			{
				// Compute difference image and error

				DIRECT_A3D_ELEM(Idiff, batchIdx, i, j) = DIRECT_A3D_ELEM(Iexp, batchIdx, i, j) - DIRECT_A3D_ELEM(Itheo, batchIdx, i, j);
				mean_error += DIRECT_A3D_ELEM(Idiff, batchIdx, i, j) * DIRECT_A3D_ELEM(Idiff, batchIdx, i, j);

				// Compute the correction image

				DIRECT_A3D_ELEM(Icorr, batchIdx, i, j) = XMIPP_MAX(DIRECT_A3D_ELEM(Icorr, batchIdx, i, j), 1);

				DIRECT_A3D_ELEM(Icorr, batchIdx, i, j) =
					this->lambdaART * DIRECT_A3D_ELEM(Idiff, batchIdx, i, j) / (batchSize * DIRECT_A3D_ELEM(Icorr, batchIdx, i, j));
			}
		}
		mean_error /= YXSIZE(Iexp);
#pragma omp for
		for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
			tmpItheo.data = &(A3D_ELEM(Itheo, batchIdx, 0, 0));
			tmpIcorr.data = &(A3D_ELEM(Icorr, batchIdx, 0, 0));

			this->project_Pseudo(tmpItheo, tmpIcorr,
				EulerVec[batchIdx], shiftX, shiftY, PSEUDO_BACKWARD);
		}
	}
	delete[] EulerVec;
	return mean_error;
}

RDOUBLE PseudoProjector::ART_batched(const MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &Itheo, MultidimArray<RDOUBLE> &Icorr, MultidimArray<RDOUBLE> &Idiff, idxtype batchSize, float3 *angles, RDOUBLE shiftX, RDOUBLE shiftY)
{
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Idiff.resize(Iexp, false);
	Matrix2D<RDOUBLE> *EulerVec = new Matrix2D<RDOUBLE>[batchSize];
	RDOUBLE mean_error = 0;
#pragma omp parallel
	{
		//Initialize array for the slice view
		MultidimArray<RDOUBLE> tmpItheo, tmpIcorr;
		tmpItheo.xdim = tmpIcorr.xdim = Iexp.xdim;
		tmpItheo.ydim = tmpIcorr.ydim = Iexp.xdim;
		tmpItheo.destroyData = tmpIcorr.destroyData = false;
		tmpItheo.yxdim = tmpItheo.nzyxdim = tmpItheo.zyxdim = tmpItheo.xdim*tmpItheo.ydim;
		tmpIcorr.yxdim = tmpIcorr.nzyxdim = tmpIcorr.zyxdim = tmpIcorr.xdim*tmpItheo.ydim;
#pragma omp for
		for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
			tmpItheo.data = &(A3D_ELEM(Itheo, batchIdx, 0, 0));
			tmpIcorr.data = &(A3D_ELEM(Icorr, batchIdx, 0, 0));
			Euler_angles2matrix(angles[batchIdx].x, angles[batchIdx].y, angles[batchIdx].z, EulerVec[batchIdx]);
			this->project_Pseudo(tmpItheo, tmpIcorr,
				EulerVec[batchIdx], shiftX, shiftY, PSEUDO_FORWARD);



			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
			{
				// Compute difference image and error

				DIRECT_A3D_ELEM(Idiff, batchIdx, i, j) = DIRECT_A3D_ELEM(Iexp, batchIdx, i, j) - DIRECT_A3D_ELEM(Itheo, batchIdx, i, j);
				mean_error += DIRECT_A3D_ELEM(Idiff, batchIdx, i, j) * DIRECT_A3D_ELEM(Idiff, batchIdx, i, j);

				// Compute the correction image

				DIRECT_A3D_ELEM(Icorr, batchIdx, i, j) = XMIPP_MAX(DIRECT_A3D_ELEM(Icorr, batchIdx, i, j), 1);

				DIRECT_A3D_ELEM(Icorr, batchIdx, i, j) =
					this->lambdaART * DIRECT_A3D_ELEM(Idiff, batchIdx, i, j) / (batchSize * DIRECT_A3D_ELEM(Icorr, batchIdx, i, j));
			}
		}
		mean_error /= YXSIZE(Iexp);
#pragma omp for
		for (int batchIdx = 0; batchIdx < batchSize; batchIdx++) {
			tmpItheo.data = &(A3D_ELEM(Itheo, batchIdx, 0, 0));
			tmpIcorr.data = &(A3D_ELEM(Icorr, batchIdx, 0, 0));

			this->project_Pseudo(tmpItheo, tmpIcorr,
				EulerVec[batchIdx], shiftX, shiftY, PSEUDO_BACKWARD);
		}
	}
	delete[] EulerVec;
	return mean_error;
}

RDOUBLE PseudoProjector::ART_multi_Image_step(RDOUBLE * Iexp, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE tableLength, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<RDOUBLE> > Images;
	RDOUBLE itError = 0.0;

	for (size_t i = 0; i < numImages; i++)
	{
		RDOUBLE * tableTmp = gaussianProjectionTable.vdata;
		RDOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;

		gaussianProjectionTable.vdata = gaussTables + i * (GAUSS_FACTOR * Dims.x / 2);
		gaussianProjectionTable2.vdata = gaussTables2 + i * (GAUSS_FACTOR * Dims.x / 2);
		double oldBorder = this->tableLength;
		this->tableLength = tableLength;
		MultidimArray<RDOUBLE> tmp = MultidimArray<RDOUBLE>(1, Dims.y, Dims.x);
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		tmp.destroyData = false;
		itError += ART_single_image(tmp, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);

		gaussianProjectionTable.vdata = tableTmp;
		gaussianProjectionTable2.vdata = tableTmp2;
		this->tableLength = oldBorder;
	}
	return itError;
}

RDOUBLE PseudoProjector::ART_multi_Image_step(RDOUBLE * Iexp, float3 * angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<RDOUBLE> > Images;
	RDOUBLE itError = 0.0;
	MultidimArray<RDOUBLE> tmp = MultidimArray<RDOUBLE>(1, Dims.y, Dims.x);
	for (size_t i = 0; i < numImages; i++)
	{

		tmp.data = Iexp + i * (Dims.x*Dims.y);

		itError += ART_single_image(tmp, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);
	}
	tmp.data = NULL;
	return itError;
}

RDOUBLE PseudoProjector::ART_multi_Image_step(std::vector< MultidimArray<RDOUBLE> > Iexp, std::vector<float3> angles, RDOUBLE shiftX, RDOUBLE shiftY) {
	RDOUBLE itError = 0.0;
	for (unsigned int i = 0; i < Iexp.size(); i++)
	{
		itError += ART_single_image(Iexp[i], angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);
	}
	return itError / Iexp.size(); // Mean Error
}

RDOUBLE PseudoProjector::ART_multi_Image_step_DB(RDOUBLE * Iexp, RDOUBLE * Itheo, RDOUBLE * Icorr, RDOUBLE * Idiff, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE tableLength, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<RDOUBLE> > Images;
	RDOUBLE itError = 0.0;

	for (size_t i = 0; i < numImages; i++)
	{
		RDOUBLE * tableTmp = gaussianProjectionTable.vdata;
		RDOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;

		gaussianProjectionTable.vdata = gaussTables + i * (GAUSS_FACTOR * Dims.x / 2);
		gaussianProjectionTable2.vdata = gaussTables2 + i * (GAUSS_FACTOR * Dims.x / 2);
		double oldBorder = this->tableLength;
		this->tableLength = tableLength;
		MultidimArray<RDOUBLE> tmp = MultidimArray<RDOUBLE>(1, Dims.y, Dims.x);
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		tmp.destroyData = false;

		MultidimArray<RDOUBLE> tmp2 = MultidimArray<RDOUBLE>(1, Dims.y, Dims.x);
		tmp2.data = Itheo + i * (Dims.x*Dims.y);
		tmp2.destroyData = false;

		MultidimArray<RDOUBLE> tmp3 = MultidimArray<RDOUBLE>(1, Dims.y, Dims.x);
		tmp3.data = Icorr + i * (Dims.x*Dims.y);
		tmp3.destroyData = false;

		MultidimArray<RDOUBLE> tmp4 = MultidimArray<RDOUBLE>(1, Dims.y, Dims.x);
		tmp4.data = Idiff + i * (Dims.x*Dims.y);
		tmp4.destroyData = false;

		itError += ART_single_image(tmp, tmp2, tmp3, tmp4, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);

		gaussianProjectionTable.vdata = tableTmp;
		gaussianProjectionTable2.vdata = tableTmp2;
		this->tableLength = oldBorder;
	}
	return itError;
}

MRCImage<RDOUBLE> *PseudoProjector::create3DImage(RDOUBLE oversampling) {

	MRCImage<RDOUBLE> *Volume = new MRCImage<RDOUBLE>();
	atoms->RasterizeToVolume(Volume->data, Dims, oversampling);
	Volume->header.dimensions = Dims;
	return Volume;
}

RDOUBLE PseudoProjector::CTFSIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY) {

	//cudaErrchk(cudaSetDevice(1));
	cudaErrchk(cudaDeviceSynchronize());

	float3 * d_ctfAtomPositions;
	cudaErrchk(cudaMalloc((void**)&d_ctfAtomPositions, ctfAtoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_ctfAtomPositions, ctfAtoms->AtomPositions.data(), ctfAtoms->NAtoms, cudaMemcpyHostToDevice));

	float * d_ctfAtomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_ctfAtomIntensities, ctfAtoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_ctfAtomIntensities, ctfAtoms->AtomWeights.data(), ctfAtoms->NAtoms, cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)2048);	//Hard limit of elementsPerBatch instead of calculating
	if (Itheo != NULL)
		Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Icorr != NULL)
		Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Idiff != NULL)
		Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (superICorr != NULL)
		superICorr->resizeNoCopy(numAngles, superDimsproj.y, superDimsproj.x);

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections;
		float * d_iExp;
		float * d_projections;
		tcomplex * d_projectionsBatchFFT;

		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_iExp, ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projectionsBatchFFT, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(tcomplex)));



		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);

			float3 * h_angles = angles + startIm;

			cudaErrchk(cudaMemcpy(d_iExp, Iexp.data + startIm * Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
			RealspacePseudoProjectForward(d_ctfAtomPositions, d_ctfAtomIntensities, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch, NULL, d_projectionsBatchFFT);

			d_IFFTC2R(d_projectionsBatchFFT, d_projectionsBatch, &planBackward);

			if (Itheo != NULL)
				cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			if (Idiff != NULL)
				cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
			if (Icorr != NULL)
				cudaErrchk(cudaMemcpy(Icorr->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}

		cudaErrchk(cudaFree(d_iExp));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cufftErrchk(cufftPlanMany(&planForward, ndims, n + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionF, ElementsPerBatch));

		cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		cufftErrchk(cufftPlanMany(&planBackward, ndims, nSuper + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionB, ElementsPerBatch));


		// Backproject
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			float3 * h_angles = angles + startIm;

			d_Scale(d_projectionsBatch, d_superProjections, gtom::toInt3(dimsproj), gtom::toInt3(superDimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (superICorr != NULL)
				cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_ctfAtomPositions, d_ctfAtomIntensities, ctfAtoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

		}
		cudaErrchk(cudaMemcpy(ctfAtoms->AtomWeights.data(), d_ctfAtomIntensities, ctfAtoms->NAtoms * sizeof(*(ctfAtoms->AtomWeights.data())), cudaMemcpyDeviceToHost));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_projections));
		cudaErrchk(cudaFree(d_projectionsBatchFFT));

	}
	cudaErrchk(cudaFree(d_ctfAtomPositions));
	cudaErrchk(cudaFree(d_ctfAtomIntensities));
	cudaErrchk(cudaSetDevice(0));
	return 0.0;

}

RDOUBLE PseudoProjector::CTFSIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, int *positionMatching, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY) {

	//cudaErrchk(cudaSetDevice(1));
	cudaErrchk(cudaDeviceSynchronize());

	float3 * d_ctfAtomPositions;
	cudaErrchk(cudaMalloc((void**)&d_ctfAtomPositions, ctfAtoms->alternativePositions.size() * ctfAtoms->NAtoms * sizeof(float3)));
	for (size_t i = 0; i < atoms->alternativePositions.size(); i++)
	{
		cudaErrchk(cudaMemcpy(d_ctfAtomPositions + i * atoms->NAtoms, ctfAtoms->alternativePositions[i], ctfAtoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));
	}

	int * d_positionMapping;
	cudaErrchk(cudaMalloc((void**)&d_positionMapping, numAngles * sizeof(*d_positionMapping)));
	cudaErrchk(cudaMemcpy(d_positionMapping, positionMatching, numAngles * sizeof(*d_positionMapping), cudaMemcpyHostToDevice));


	float * d_ctfAtomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_ctfAtomIntensities, ctfAtoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_ctfAtomIntensities, ctfAtoms->AtomWeights.data(), ctfAtoms->NAtoms, cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)2048);	//Hard limit of elementsPerBatch instead of calculating
	if (Itheo != NULL)
		Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Icorr != NULL)
		Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Idiff != NULL)
		Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (superICorr != NULL)
		superICorr->resizeNoCopy(numAngles, superDimsproj.y, superDimsproj.x);

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections;
		float * d_iExp;
		float * d_projections;
		tcomplex * d_projectionsBatchFFT;

		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_iExp, ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projectionsBatchFFT, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(tcomplex)));



		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);

			float3 * h_angles = angles + startIm;

			cudaErrchk(cudaMemcpy(d_iExp, Iexp.data + startIm * Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
			RealspacePseudoProjectForward(d_ctfAtomPositions, d_ctfAtomIntensities, d_positionMapping, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch, NULL, d_projectionsBatchFFT);

			d_IFFTC2R(d_projectionsBatchFFT, d_projectionsBatch, &planBackward);

			if (Itheo != NULL)
				cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			if (Idiff != NULL)
				cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
			if (Icorr != NULL)
				cudaErrchk(cudaMemcpy(Icorr->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}

		cudaErrchk(cudaFree(d_iExp));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cufftErrchk(cufftPlanMany(&planForward, ndims, n + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionF, ElementsPerBatch));

		cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		cufftErrchk(cufftPlanMany(&planBackward, ndims, nSuper + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionB, ElementsPerBatch));


		// Backproject
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			float3 * h_angles = angles + startIm;

			d_Scale(d_projectionsBatch, d_superProjections, gtom::toInt3(dimsproj), gtom::toInt3(superDimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (superICorr != NULL)
				cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_ctfAtomPositions, d_ctfAtomIntensities, d_positionMapping, ctfAtoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

		}
		cudaErrchk(cudaMemcpy(ctfAtoms->AtomWeights.data(), d_ctfAtomIntensities, ctfAtoms->NAtoms * sizeof(*(ctfAtoms->AtomWeights.data())), cudaMemcpyDeviceToHost));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_projections));
		cudaErrchk(cudaFree(d_projectionsBatchFFT));

	}
	cudaErrchk(cudaFree(d_ctfAtomPositions));
	cudaErrchk(cudaFree(d_ctfAtomIntensities));
	cudaErrchk(cudaSetDevice(0));
	return 0.0;

}

void PseudoProjector::projectForward(float3 *angles, int* positionMatching, MultidimArray<RDOUBLE>* CTFs, MultidimArray<RDOUBLE>& projections, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY)
{
	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->alternativePositions.size() * atoms->NAtoms * sizeof(float3)));
	for (size_t i = 0; i < atoms->alternativePositions.size(); i++)
	{
		cudaErrchk(cudaMemcpy(d_atomPositions + i * atoms->NAtoms, atoms->alternativePositions[i], atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));
	}

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	int * d_positionMatching;
	cudaErrchk(cudaMalloc((void**)&d_positionMatching, numAngles * sizeof(*d_positionMatching)));
	cudaErrchk(cudaMemcpy(d_positionMatching, positionMatching, numAngles * sizeof(*d_positionMatching), cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)2048);	//Hard limit of elementsPerBatch instead of calculating

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections;
		float * d_projections;
		float * d_ctfs;
		tcomplex * d_projectionsBatchFFT;

		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_ctfs, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projectionsBatchFFT, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(tcomplex)));

		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);

			float3 * h_angles = angles + startIm;

			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, d_positionMatching, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);

			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch, NULL, d_projectionsBatchFFT);

			if (CTFs != NULL) {
				cudaErrchk(cudaMemcpy(d_ctfs, CTFs->data + startIm * ElementsFFT2(dimsproj), batch*ElementsFFT2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
				d_ComplexMultiplyByVector(d_projectionsBatchFFT, d_ctfs, d_projectionsBatchFFT, batch*ElementsFFT2(dimsproj), 1);
			}
			d_IFFTC2R(d_projectionsBatchFFT, d_projectionsBatch, DimensionCount(gtom::toInt3(dimsproj)), gtom::toInt3(dimsproj), batch);


		}
		projections.resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
		cudaErrchk(cudaMemcpy(projections.data, d_projections, projections.nzyxdim*sizeof(*d_projections),cudaMemcpyDeviceToHost));
		cudaErrchk(cudaFree(d_projections));
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_ctfs));
		cudaErrchk(cudaFree(d_projectionsBatchFFT));
		cudaErrchk(cudaFree(d_atomIntensities));
		cudaErrchk(cudaFree(d_atomPositions));
		cudaErrchk(cudaFree(d_positionMatching));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);
	}
}

void PseudoProjector::createMask(float3 *angles, int* positionMatching, MultidimArray<RDOUBLE>* CTFs, MultidimArray<RDOUBLE>& projections, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY)
{
	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->alternativePositions.size() * atoms->NAtoms * sizeof(float3)));
	for (size_t i = 0; i < atoms->alternativePositions.size(); i++)
	{
		cudaErrchk(cudaMemcpy(d_atomPositions + i * atoms->NAtoms, atoms->alternativePositions[i], atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));
	}

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	int * d_positionMatching;
	cudaErrchk(cudaMalloc((void**)&d_positionMatching, numAngles * sizeof(*d_positionMatching)));
	cudaErrchk(cudaMemcpy(d_positionMatching, positionMatching, numAngles * sizeof(*d_positionMatching), cudaMemcpyHostToDevice));

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int2 dimsproj = { Dims.x,  Dims.y };


	float * d_projections;
	cudaErrchk(cudaMalloc(&d_projections, numAngles * Elements2(dimsproj) * sizeof(float)));


	// Forward project


	RealspacePseudoCreateMask(d_atomPositions, d_atomIntensities, d_positionMatching, atoms->NAtoms, dimsvolume, d_projections, dimsproj, angles, numAngles);
	cudaErrchk(cudaPeekAtLastError());

	projections.resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	cudaErrchk(cudaMemcpy(projections.data, d_projections, projections.nzyxdim * sizeof(*d_projections), cudaMemcpyDeviceToHost));
	cudaErrchk(cudaFree(d_projections));
	cudaErrchk(cudaFree(d_atomIntensities));
	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_positionMatching));
}

void PseudoProjector::project_Pseudo(RDOUBLE * out, RDOUBLE * out_nrm,float3 angles, RDOUBLE shiftX, RDOUBLE shiftY, int direction)
{
	MultidimArray<RDOUBLE> proj = MultidimArray<RDOUBLE>(1, this->Dims.y, this->Dims.x);
	proj.data = out;
	proj.xinit = shiftX;
	proj.yinit = shiftY;
	proj.destroyData = false;

	MultidimArray<RDOUBLE> proj_nrm = MultidimArray<RDOUBLE>(1, this->Dims.y, this->Dims.x);
	if (out_nrm != NULL)
	{
		proj_nrm.data = out_nrm;
		proj_nrm.destroyData = NULL;
	}
	proj_nrm.xinit = shiftX;
	proj_nrm.yinit = shiftY;

	Matrix2D<RDOUBLE> EulerMat = Matrix2D<RDOUBLE>();
	Euler_angles2matrix(angles.x, angles.y, angles.z, EulerMat);


	this->project_Pseudo(proj, proj_nrm, EulerMat, shiftX, shiftY, direction);
}

void PseudoProjector::project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj, std::vector<Matrix1D<RDOUBLE>> * atomPositions, Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY, int direction) {
	// Project all pseudo atoms ............................................
	int numAtoms = (*atomPositions).size();
	Matrix1D<RDOUBLE> actprj(3);
	double sigma4 = this->tableLength;
	Matrix1D<RDOUBLE> actualAtomPosition;

	for (int n = 0; n < numAtoms; n++)
	{
		actualAtomPosition = (*atomPositions)[n]*super;
		XX(actualAtomPosition) -= Dims.x / 2;
		YY(actualAtomPosition) -= Dims.y / 2;
		ZZ(actualAtomPosition) -= Dims.z / 2;

		RDOUBLE weight = atoms->AtomWeights[n];

		Uproject_to_plane(actualAtomPosition, Euler, actprj);
		XX(actprj) += Dims.x / 2;
		YY(actprj) += Dims.y / 2;
		ZZ(actprj) += Dims.z / 2;

		XX(actprj) += shiftX;
		YY(actprj) += shiftY;
		if (mode == ATOM_INTERPOLATE) {
			if (direction == PSEUDO_FORWARD) {
				int X0 = (int)XX(actprj);
				RDOUBLE ix = XX(actprj) - X0;
				int X1 = X0 + 1;

				int Y0 = (int)YY(actprj);
				RDOUBLE iy = YY(actprj) - Y0;
				int Y1 = Y0 + 1;

				RDOUBLE v0 = 1.0f - iy;
				RDOUBLE v1 = iy;

				RDOUBLE v00 = (1.0f - ix) * v0;
				RDOUBLE v10 = ix * v0;
				RDOUBLE v01 = (1.0f - ix) * v1;
				RDOUBLE v11 = ix * v1;

				A2D_ELEM(proj, Y0, X0) += weight * v00;
				A2D_ELEM(proj, Y0, X1) += weight * v01;
				A2D_ELEM(proj, Y1, X0) += weight * v10;
				A2D_ELEM(proj, Y1, X1) += weight * v11;
			}
			else if (direction == PSEUDO_BACKWARD) {

				int X0 = (int)XX(actprj);
				RDOUBLE ix = XX(actprj) - X0;
				int X1 = X0 + 1;

				int Y0 = (int)YY(actprj);
				RDOUBLE iy = YY(actprj) - Y0;
				int Y1 = Y0 + 1;

				RDOUBLE v00 = A3D_ELEM(norm_proj, 0, Y0, X0);
				RDOUBLE v01 = A3D_ELEM(norm_proj, 0, Y0, X1);
				RDOUBLE v10 = A3D_ELEM(norm_proj, 0, Y1, X0);
				RDOUBLE v11 = A3D_ELEM(norm_proj, 0, Y1, X1);


				RDOUBLE v0 = Lerp(v00, v01, ix);
				RDOUBLE v1 = Lerp(v10, v11, ix);

				RDOUBLE v = Lerp(v0, v1, iy);

			}
		}
		//Gaussian projection mode
		// Search for integer corners for this basis
		
		else if (mode == ATOM_GAUSSIAN) {
			int XX_corner1 = CEIL(XMIPP_MAX(STARTINGX(proj), XX(actprj) - sigma4));
			int YY_corner1 = CEIL(XMIPP_MAX(STARTINGY(proj), YY(actprj) - sigma4));
			int XX_corner2 = FLOOR(XMIPP_MIN(FINISHINGX(proj), XX(actprj) + sigma4));
			int YY_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(proj), YY(actprj) + sigma4));

			// Check if the basis falls outside the projection plane
			if (XX_corner1 <= XX_corner2 && YY_corner1 <= YY_corner2)
			{
				RDOUBLE vol_corr = 0;

				// Effectively project this basis
				for (int y = YY_corner1; y <= YY_corner2; y++)
				{
					RDOUBLE y_diff2 = y - YY(actprj);
					y_diff2 = y_diff2 * y_diff2;
					for (int x = XX_corner1; x <= XX_corner2; x++)
					{
						RDOUBLE x_diff2 = x - XX(actprj);
						x_diff2 = x_diff2 * x_diff2;
						RDOUBLE r = sqrt(x_diff2 + y_diff2);
						RDOUBLE didx = r * GAUSS_FACTOR;
						int idx = ROUND(didx);
						RDOUBLE a = VEC_ELEM(gaussianProjectionTable, idx);
						RDOUBLE a2 = VEC_ELEM(gaussianProjectionTable2, idx);

						if (a < 0 || a>2) {
							bool is = true;
						}
						if (direction == PSEUDO_FORWARD)
						{
							A2D_ELEM(proj, y, x) += weight * a;
							A2D_ELEM(norm_proj, y, x) += a2;

						}
						else
						{
							vol_corr += A2D_ELEM(norm_proj, y, x) * a;

						}
					}
				}

				if (direction == PSEUDO_BACKWARD)
				{
					atoms->AtomWeights[n] += vol_corr;
					//atomWeight[n] = atomWeight[n] > 0 ? atomWeight[n] : 0;
				}
			}

		} // If not collapsed
		
	}
}

 /** Projection of a pseudoatom volume */
void PseudoProjector::project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj, Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY, int direction)
{
	// Project all pseudo atoms ............................................
	int numAtoms = atoms->AtomPositions.size();
	Matrix1D<RDOUBLE> actprj(3);
	double sigma4 = this->tableLength;
	Matrix1D<RDOUBLE> actualAtomPosition;

	for (int n = 0; n < numAtoms; n++)
	{

		XX(actualAtomPosition) = atoms->AtomPositions[n].x - Dims.x / 2;
		YY(actualAtomPosition) = atoms->AtomPositions[n].y - Dims.y / 2;
		ZZ(actualAtomPosition) = atoms->AtomPositions[n].z - Dims.z / 2;

		RDOUBLE weight = atoms->AtomWeights[n];

		Uproject_to_plane(actualAtomPosition, Euler, actprj);
		XX(actprj) += Dims.x / 2;
		YY(actprj) += Dims.y / 2;
		ZZ(actprj) += Dims.z / 2;

		XX(actprj) += shiftX;
		YY(actprj) += shiftY;

		if (mode == ATOM_GAUSSIAN)
		{

			// Search for integer corners for this basis
			int XX_corner1 = CEIL(XMIPP_MAX(STARTINGX(proj), XX(actprj) - sigma4));
			int YY_corner1 = CEIL(XMIPP_MAX(STARTINGY(proj), YY(actprj) - sigma4));
			int XX_corner2 = FLOOR(XMIPP_MIN(FINISHINGX(proj), XX(actprj) + sigma4));
			int YY_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(proj), YY(actprj) + sigma4));

			// Check if the basis falls outside the projection plane
			if (XX_corner1 <= XX_corner2 && YY_corner1 <= YY_corner2)
			{
				RDOUBLE vol_corr = 0;

				// Effectively project this basis
				for (int y = YY_corner1; y <= YY_corner2; y++)
				{
					RDOUBLE y_diff2 = y - YY(actprj);
					y_diff2 = y_diff2 * y_diff2;
					for (int x = XX_corner1; x <= XX_corner2; x++)
					{
						RDOUBLE x_diff2 = x - XX(actprj);
						x_diff2 = x_diff2 * x_diff2;
						RDOUBLE r = sqrt(x_diff2 + y_diff2);
						RDOUBLE didx = r * GAUSS_FACTOR;
						int idx = ROUND(didx);
						RDOUBLE a = VEC_ELEM(gaussianProjectionTable, idx);
						RDOUBLE a2 = VEC_ELEM(gaussianProjectionTable2, idx);

						if (a < 0 || a>2) {
							bool is = true;
						}
						if (direction == PSEUDO_FORWARD)
						{
							A2D_ELEM(proj, y, x) += weight * a;
							A2D_ELEM(norm_proj, y, x) += a2;

						}
						else
						{
							vol_corr += A2D_ELEM(norm_proj, y, x) * a;

						}
					}
				}

				if (direction == PSEUDO_BACKWARD)
				{
					atoms->AtomWeights[n] += vol_corr;
					//atomWeight[n] = atomWeight[n] > 0 ? atomWeight[n] : 0;
				}
			} // If not collapsed
		}
		else if (mode == ATOM_INTERPOLATE) {
			if (direction == PSEUDO_FORWARD) {
				int X0 = (int)XX(actprj);
				RDOUBLE ix = XX(actprj) - X0;
				int X1 = X0 + 1;

				int Y0 = (int)YY(actprj);
				RDOUBLE iy = YY(actprj) - Y0;
				int Y1 = Y0 + 1;

				RDOUBLE v0 = 1.0f - iy;
				RDOUBLE v1 = iy;

				RDOUBLE v00 = (1.0f - ix) * v0;
				RDOUBLE v10 = ix * v0;
				RDOUBLE v01 = (1.0f - ix) * v1;
				RDOUBLE v11 = ix * v1;

				A2D_ELEM(proj, Y0, X0) += weight * v00;
				A2D_ELEM(proj, Y0, X1) += weight * v01;
				A2D_ELEM(proj, Y1, X0) += weight * v10;
				A2D_ELEM(proj, Y1, X1) += weight * v11;
			}
			else if (direction == PSEUDO_BACKWARD) {

				int X0 = (int)XX(actprj);
				RDOUBLE ix = XX(actprj) - X0;
				int X1 = X0 + 1;

				int Y0 = (int)YY(actprj);
				RDOUBLE iy = YY(actprj) - Y0;
				int Y1 = Y0 + 1;

				RDOUBLE v00 = A3D_ELEM(norm_proj, 0, Y0, X0);
				RDOUBLE v01 = A3D_ELEM(norm_proj, 0, Y0, X1);
				RDOUBLE v10 = A3D_ELEM(norm_proj, 0, Y1, X0);
				RDOUBLE v11 = A3D_ELEM(norm_proj, 0, Y1, X1);


				RDOUBLE v0 = Lerp(v00, v01, ix);
				RDOUBLE v1 = Lerp(v10, v11, ix);

				RDOUBLE v = Lerp(v0, v1, iy);

			}
		}
		else
			REPORT_ERROR(std::string("This projection type is not implemented ") + __FILE__ + ": " + std::to_string(__LINE__));

	}

}

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY)
{

	return SIRT(Iexp, angles, numAngles, NULL, NULL, NULL, NULL, shiftX, shiftY);
}

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY)
{
	//cudaErrchk(cudaSetDevice(1));
	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, atoms->AtomPositions.data(), atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = {Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2*Elements2(superDimsproj)+ Elements2(dimsproj)+ Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)1024);	//Hard limit of elementsPerBatch instead of calculating
	if (Itheo != NULL)
		Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Icorr != NULL)
		Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Idiff != NULL)
		Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (superICorr != NULL)
		superICorr->resizeNoCopy(numAngles, superDimsproj.y, superDimsproj.x);

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = {1, superDimsproj.y, superDimsproj.x };
	int n[3] = {1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections; 	
		float * d_iExp;
		float * d_projections;

			cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
			cudaErrchk(cudaMalloc(&d_iExp, ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
			cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));

		

		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm+=ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);


			float3 * h_angles = angles+startIm;
			cudaMemcpy(d_iExp, Iexp.data+startIm*Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice);
			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if(Itheo != NULL)
				cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			if (Idiff != NULL)
				cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
			if (Icorr != NULL)
				cudaErrchk(cudaMemcpy(Icorr->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}
		
		cudaErrchk(cudaFree(d_iExp));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cufftErrchk(cufftPlanMany(&planForward, ndims, n + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionF, ElementsPerBatch));

		cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		cufftErrchk(cufftPlanMany(&planBackward, ndims, nSuper + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionB, ElementsPerBatch));


		// Backproject
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			float3 * h_angles = angles + startIm;

			d_Scale(d_projectionsBatch, d_superProjections, gtom::toInt3(dimsproj), gtom::toInt3(superDimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (superICorr != NULL)
				cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_atomPositions, d_atomIntensities, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());
			
		}
		cudaErrchk(cudaMemcpy(atoms->AtomWeights.data(), d_atomIntensities, atoms->NAtoms*sizeof(*(atoms->AtomWeights.data())), cudaMemcpyDeviceToHost));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_projections));

	}
	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_atomIntensities));
	return 0.0;
}

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, int *positionMatching, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY)
{
	//cudaErrchk(cudaSetDevice(1));
	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->alternativePositions.size() * atoms->NAtoms * sizeof(float3)));
	for (size_t i = 0; i < atoms->alternativePositions.size(); i++)
	{
		cudaErrchk(cudaMemcpy(d_atomPositions+i* atoms->NAtoms, atoms->alternativePositions[i], atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));
	}
	
	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	int * d_positionMapping;
	cudaErrchk(cudaMalloc((void**)&d_positionMapping, numAngles * sizeof(*d_positionMapping)));
	cudaErrchk(cudaMemcpy(d_positionMapping, positionMatching, numAngles * sizeof(*d_positionMapping), cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)512);	//Hard limit of elementsPerBatch instead of calculating
	if (Itheo != NULL)
		Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Icorr != NULL)
		Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Idiff != NULL)
		Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (superICorr != NULL)
		superICorr->resizeNoCopy(numAngles, superDimsproj.y, superDimsproj.x);

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections;
		float * d_iExp;
		float * d_projections;

		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_iExp, ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));



		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			int * d_positionMappingBatch = d_positionMapping + startIm;

			float3 * h_angles = angles + startIm;
			cudaMemcpy(d_iExp, Iexp.data + startIm * Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice);
			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, d_positionMappingBatch, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (Itheo != NULL)
				cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			if (Idiff != NULL)
				cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
			if (Icorr != NULL)
				cudaErrchk(cudaMemcpy(Icorr->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}

		cudaErrchk(cudaFree(d_iExp));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cufftErrchk(cufftPlanMany(&planForward, ndims, n + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionF, ElementsPerBatch));

		cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		cufftErrchk(cufftPlanMany(&planBackward, ndims, nSuper + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionB, ElementsPerBatch));


		// Backproject
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			int * d_positionMappingBatch = d_positionMapping + startIm;
			float3 * h_angles = angles + startIm;

			d_Scale(d_projectionsBatch, d_superProjections, gtom::toInt3(dimsproj), gtom::toInt3(superDimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (superICorr != NULL)
				cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_atomPositions, d_atomIntensities, d_positionMappingBatch, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

		}
		cudaErrchk(cudaMemcpy(atoms->AtomWeights.data(), d_atomIntensities, atoms->NAtoms * sizeof(*(atoms->AtomWeights.data())), cudaMemcpyDeviceToHost));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_projections));

	}
	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_atomIntensities));
	cudaErrchk(cudaSetDevice(0));
	return 0.0;
}

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &CTFs, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY)
{
	MultidimArray<RDOUBLE> realCTF;
	int it = 0;
	realspaceCTF(CTFs, realCTF, this->Dims);

	this->CTFSIRT(realCTF, angles, numAngles, NULL, NULL, NULL, NULL, 0, 0);

	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, atoms->AtomPositions.data(), atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	float * d_ctfAtomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_ctfAtomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_ctfAtomIntensities, ctfAtoms->AtomWeights.data(), atoms->NAtoms, cudaMemcpyHostToDevice));

	FileName tmpDir = std::string("D:\\EMD\\9233\\TomoReconstructions\\Debug\\") + std::to_string(it) + "_";

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	int3 dimIExp = { Iexp.xdim, Iexp.ydim, Iexp.zdim };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)2048);	//Hard limit of elementsPerBatch instead of calculating
	if (Itheo != NULL)
		Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Icorr != NULL)
		Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Idiff != NULL)
		Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (superICorr != NULL)
		superICorr->resizeNoCopy(numAngles, superDimsproj.y, superDimsproj.x);

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections;
		float * d_iExp;
		float * d_projections;
		float * d_ctfs;
		tcomplex * d_projectionsBatchFFT;

		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_iExp, ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_ctfs, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projectionsBatchFFT, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(tcomplex)));



		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			int3 batchProjDim = dimIExp;
			batchProjDim.z = batch;

			int3 batchSuperProjDim = dimIExp*super;
			batchSuperProjDim.z = batch;
			float3 * h_angles = angles + startIm;

			cudaErrchk(cudaMemcpy(d_iExp, Iexp.data + startIm * Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
			if (true)
				outputDeviceAsImage(d_iExp, batchProjDim, tmpDir + std::string("d_iExp_it") + std::to_string(startIm) + ".mrc", false);

			cudaErrchk(cudaMemcpy(d_ctfs, CTFs.data + startIm * ElementsFFT2(dimsproj), batch*ElementsFFT2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			if (true)
				outputDeviceAsImage(d_superProjections, batchSuperProjDim, tmpDir + std::string("d_superProjectionsBatch_it") + std::to_string(startIm) + ".mrc", false);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch, NULL, d_projectionsBatchFFT);
			if (true)
				outputDeviceAsImage(d_projectionsBatch, batchProjDim, tmpDir + std::string("d_projectionsBatch_it") + std::to_string(startIm) + ".mrc", false);
			d_ComplexMultiplyByVector(d_projectionsBatchFFT, d_ctfs, d_projectionsBatchFFT, batch*ElementsFFT2(dimsproj), 1);
			d_IFFTC2R(d_projectionsBatchFFT, d_projectionsBatch, DimensionCount(gtom::toInt3(dimsproj)), gtom::toInt3(dimsproj), batch);
			if (true)
				outputDeviceAsImage(d_projectionsBatch, batchProjDim, tmpDir + std::string("d_projectionsBatchConvolved_it") + std::to_string(startIm) + ".mrc", false);
			if (Itheo != NULL)
				cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			if (Idiff != NULL)
				cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
 			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
			if (Icorr != NULL)
				cudaErrchk(cudaMemcpy(Icorr->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}

		cudaErrchk(cudaFree(d_iExp));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cufftErrchk(cufftPlanMany(&planForward, ndims, n + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionF, ElementsPerBatch));

		cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		cufftErrchk(cufftPlanMany(&planBackward, ndims, nSuper + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionB, ElementsPerBatch));


		float* d_copyAtomIntensities;
		cudaErrchk(cudaMalloc(&d_copyAtomIntensities, atoms->NAtoms * sizeof(*d_copyAtomIntensities)));
		cudaErrchk(cudaMemcpy(d_copyAtomIntensities, d_atomIntensities, atoms->NAtoms * sizeof(*d_copyAtomIntensities), cudaMemcpyDeviceToDevice));
		// Backproject
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			float3 * h_angles = angles + startIm;

			d_Scale(d_projectionsBatch, d_superProjections, gtom::toInt3(dimsproj), gtom::toInt3(superDimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (superICorr != NULL)
				cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_atomPositions, d_atomIntensities, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

		}
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_projections));
		cudaErrchk(cudaFree(d_projectionsBatchFFT));

		MultidimArray<RDOUBLE> volBefore;
		this->atoms->RasterizeToVolume(volBefore, this->Dims, this->super, false);
		cudaErrchk(cudaMemcpy(atoms->AtomWeights.data(), d_atomIntensities, atoms->NAtoms * sizeof(*(atoms->AtomWeights.data())), cudaMemcpyDeviceToHost));
		if(true)
			outputAsImage(volBefore, tmpDir + "volBefore.mrc");


		MultidimArray<RDOUBLE> volAfter;
		this->atoms->RasterizeToVolume(volAfter, this->Dims, this->super, false);
		if (true)
			outputAsImage(volAfter, tmpDir + "volAfter.mrc");

		MultidimArray<RDOUBLE> diffConvolved = volAfter - volBefore;
		if (true)
			outputAsImage(diffConvolved, tmpDir + "volDiff.mrc");

		float *d_diffConvolved;
		cudaErrchk(cudaMalloc(&d_diffConvolved, Elements(Dims*this->super) * sizeof(*d_diffConvolved)));
		cudaErrchk(cudaMemcpy(d_diffConvolved, diffConvolved.data, Elements(Dims*this->super) * sizeof(*d_diffConvolved), cudaMemcpyHostToDevice));
		if (true)
			outputDeviceAsImage(d_diffConvolved, this->Dims*this->super, tmpDir + "d_diffConvolved.mrc");
		tcomplex *d_fftDiffConvolved;
		cudaErrchk(cudaMalloc(&d_fftDiffConvolved, ElementsFFT(Dims*this->super) * sizeof(*d_fftDiffConvolved)));
		d_FFTR2C(d_diffConvolved, d_fftDiffConvolved, DimensionCount(this->Dims*this->super), this->Dims*this->super);
		if (true)
			outputDeviceAsImage(d_fftDiffConvolved, this->Dims * this->super, tmpDir + "d_fftDiffConvolved.mrc");

		MultidimArray<RDOUBLE> ctfRecon;
		this->ctfAtoms->RasterizeToVolume(ctfRecon, this->Dims, this->super, false);
		if (true)
			outputAsImage(ctfRecon, tmpDir + "ctfRecon.mrc");


		float *d_CTFRecon;
		cudaErrchk(cudaMalloc(&d_CTFRecon, Elements(Dims*this->super) * sizeof(*d_CTFRecon)));
		cudaErrchk(cudaMemcpy(d_CTFRecon, ctfRecon.data, Elements(Dims*this->super) * sizeof(*d_CTFRecon), cudaMemcpyHostToDevice));

		tcomplex *d_fftCTFRecon;
		cudaErrchk(cudaMalloc(&d_fftCTFRecon, ElementsFFT(Dims*this->super) * sizeof(*d_fftCTFRecon)));
		if (true)
			outputDeviceAsImage(d_fftCTFRecon, this->Dims * this->super, tmpDir + "d_fftCTFRecon.mrc");

		tfloat *d_absfftCTFRecon;
		cudaErrchk(cudaMalloc(&d_absfftCTFRecon, ElementsFFT(Dims*this->super) * sizeof(*d_absfftCTFRecon)));
		d_FFTR2C(d_CTFRecon, d_fftCTFRecon, DimensionCount(this->Dims), this->Dims*this->super);
		d_Abs(d_fftCTFRecon, d_absfftCTFRecon, ElementsFFT(Dims*this->super));
		if (true)
			outputDeviceAsImage(d_absfftCTFRecon, this->Dims * this->super, tmpDir + "d_absfftCTFRecon.mrc", true);
		d_MaxOp(d_absfftCTFRecon, 1e-2, d_absfftCTFRecon, ElementsFFT(Dims*this->super));
		d_ComplexDivideByVector(d_fftDiffConvolved, d_absfftCTFRecon, d_fftDiffConvolved, ElementsFFT(this->Dims*this->super), 1);
		if (true)
			outputDeviceAsImage(d_fftDiffConvolved, this->Dims * this->super, tmpDir + "d_fftDiffDivided.mrc");
		d_IFFTC2R(d_fftDiffConvolved, d_diffConvolved, DimensionCount(this->Dims), this->Dims*this->super, 1);
		if (true)
			outputDeviceAsImage(d_diffConvolved, this->Dims * this->super, tmpDir + "d_diffDivided.mrc");
		RealspaceVolumeUpdate(d_atomPositions, d_copyAtomIntensities, atoms->NAtoms, d_diffConvolved, this->Dims, this->super);

		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cudaErrchk(cudaFree(d_copyAtomIntensities));
		cudaErrchk(cudaFree(d_ctfs));
		cudaErrchk(cudaFree(d_diffConvolved));
		cudaErrchk(cudaFree(d_fftDiffConvolved));
		cudaErrchk(cudaFree(d_CTFRecon));
		cudaErrchk(cudaFree(d_fftCTFRecon));
		cudaErrchk(cudaFree(d_absfftCTFRecon));


	}
	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_atomIntensities));
	cudaErrchk(cudaFree(d_ctfAtomIntensities));
	return 0.0;
}

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &CTFs, float3 *angles, int *positionMatching, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY)
{
	MultidimArray<RDOUBLE> realCTF;
	int it = 0;
	realspaceCTF(CTFs, realCTF, this->Dims);

	this->CTFSIRT(realCTF, angles, positionMatching, numAngles, NULL, NULL, NULL, NULL, 0, 0);

	cudaErrchk(cudaDeviceSynchronize());
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->alternativePositions.size() * atoms->NAtoms * sizeof(float3)));
	for (size_t i = 0; i < atoms->alternativePositions.size(); i++)
	{
		cudaErrchk(cudaMemcpy(d_atomPositions + i * atoms->NAtoms, atoms->alternativePositions[i], atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));
	}

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	float * d_ctfAtomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_ctfAtomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_ctfAtomIntensities, ctfAtoms->AtomWeights.data(), atoms->NAtoms, cudaMemcpyHostToDevice));

	int * d_positionMatching;
	cudaErrchk(cudaMalloc((void**)&d_positionMatching, numAngles * sizeof(*d_positionMatching)));
	cudaErrchk(cudaMemcpy(d_positionMatching, positionMatching, numAngles * sizeof(*d_positionMatching), cudaMemcpyHostToDevice));

	FileName tmpDir = std::string("D:\\EMD\\9233\\Movement_Analysis_tomo\\800k\\Reconstruction_Single_1_real_CTF_with_weighting\\lb_0.100000_snr_inf\\Debug\\") + std::to_string(it) + "_";
	bool writeDebug = true;
	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = { Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	int3 dimIExp = { Iexp.xdim, Iexp.ydim, Iexp.zdim };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2 * Elements2(superDimsproj) + Elements2(dimsproj) + Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)2048);	//Hard limit of elementsPerBatch instead of calculating
	if (Itheo != NULL)
		Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Icorr != NULL)
		Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (Idiff != NULL)
		Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	if (superICorr != NULL)
		superICorr->resizeNoCopy(numAngles, superDimsproj.y, superDimsproj.x);

	RDOUBLE mean_error = 0.0;
	int ndims = DimensionCount(gtom::toInt3(superDimsproj));
	int nSuper[3] = { 1, superDimsproj.y, superDimsproj.x };
	int n[3] = { 1, dimsproj.y, dimsproj.x };
	cufftHandle planForward, planBackward;
	cufftType directionF = IS_TFLOAT_DOUBLE ? CUFFT_D2Z : CUFFT_R2C;

	cufftErrchk(cufftPlanMany(&planForward, ndims, nSuper + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionF, ElementsPerBatch));

	cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
	cufftErrchk(cufftPlanMany(&planBackward, ndims, n + (3 - ndims),
		NULL, 1, 0,
		NULL, 1, 0,
		directionB, ElementsPerBatch));

	{

		idxtype numBatches = (int)(std::ceil(((float)numAngles) / ((float)ElementsPerBatch)));

		float * d_superProjections;
		float * d_iExp;
		float * d_projections;
		float * d_ctfs;
		tcomplex * d_projectionsBatchFFT;

		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_iExp, ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_ctfs, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(float)));
		cudaErrchk(cudaMalloc(&d_projectionsBatchFFT, ElementsPerBatch * ElementsFFT2(dimsproj) * sizeof(tcomplex)));



		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			int3 batchProjDim = dimIExp;
			batchProjDim.z = batch;

			int3 batchSuperProjDim = dimIExp * super;
			batchSuperProjDim.z = batch;
			float3 * h_angles = angles + startIm;

			cudaErrchk(cudaMemcpy(d_iExp, Iexp.data + startIm * Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
			if (writeDebug)
				outputDeviceAsImage(d_iExp, batchProjDim, tmpDir + std::string("d_iExp_it") + std::to_string(startIm) + ".mrc", false);

			cudaErrchk(cudaMemcpy(d_ctfs, CTFs.data + startIm * ElementsFFT2(dimsproj), batch*ElementsFFT2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice));
			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, d_positionMatching, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			if (writeDebug)
				outputDeviceAsImage(d_superProjections, batchSuperProjDim, tmpDir + std::string("d_superProjectionsBatch_it") + std::to_string(startIm) + ".mrc", false);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch, NULL, d_projectionsBatchFFT);
			if (writeDebug)
				outputDeviceAsImage(d_projectionsBatch, batchProjDim, tmpDir + std::string("d_projectionsBatch_it") + std::to_string(startIm) + ".mrc", false);
			d_ComplexMultiplyByVector(d_projectionsBatchFFT, d_ctfs, d_projectionsBatchFFT, batch*ElementsFFT2(dimsproj), 1);
			d_IFFTC2R(d_projectionsBatchFFT, d_projectionsBatch, DimensionCount(gtom::toInt3(dimsproj)), gtom::toInt3(dimsproj), batch);
			if (writeDebug)
				outputDeviceAsImage(d_projectionsBatch, batchProjDim, tmpDir + std::string("d_projectionsBatchConvolved_it") + std::to_string(startIm) + ".mrc", false);
			if (Itheo != NULL)
				cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			if (Idiff != NULL)
				cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
			if (Icorr != NULL)
				cudaErrchk(cudaMemcpy(Icorr->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}

		cudaErrchk(cudaFree(d_iExp));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cufftErrchk(cufftPlanMany(&planForward, ndims, n + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionF, ElementsPerBatch));

		cufftType directionB = IS_TFLOAT_DOUBLE ? CUFFT_Z2D : CUFFT_C2R;
		cufftErrchk(cufftPlanMany(&planBackward, ndims, nSuper + (3 - ndims),
			NULL, 1, 0,
			NULL, 1, 0,
			directionB, ElementsPerBatch));


		float* d_copyAtomIntensities;
		cudaErrchk(cudaMalloc(&d_copyAtomIntensities, atoms->NAtoms * sizeof(*d_copyAtomIntensities)));
		cudaErrchk(cudaMemcpy(d_copyAtomIntensities, d_atomIntensities, atoms->NAtoms * sizeof(*d_copyAtomIntensities), cudaMemcpyDeviceToDevice));
		// Backproject
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);
			float3 * h_angles = angles + startIm;

			d_Scale(d_projectionsBatch, d_superProjections, gtom::toInt3(dimsproj), gtom::toInt3(superDimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			if (superICorr != NULL)
				cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_atomPositions, d_atomIntensities, d_positionMatching, atoms->NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

		}
		cudaErrchk(cudaFree(d_superProjections));
		cudaErrchk(cudaFree(d_projections));
		cudaErrchk(cudaFree(d_projectionsBatchFFT));

		MultidimArray<RDOUBLE> volBefore;
		this->atoms->RasterizeToVolume(volBefore, this->Dims, this->super, false);
		cudaErrchk(cudaMemcpy(atoms->AtomWeights.data(), d_atomIntensities, atoms->NAtoms * sizeof(*(atoms->AtomWeights.data())), cudaMemcpyDeviceToHost));
		if (writeDebug)
			outputAsImage(volBefore, tmpDir + "volBefore.mrc");


		MultidimArray<RDOUBLE> volAfter;
		this->atoms->RasterizeToVolume(volAfter, this->Dims, this->super, false);
		if (writeDebug)
			outputAsImage(volAfter, tmpDir + "volAfter.mrc");

		MultidimArray<RDOUBLE> diffConvolved = volAfter - volBefore;
		if (writeDebug)
			outputAsImage(diffConvolved, tmpDir + "volDiff.mrc");

		float *d_diffConvolved;
		cudaErrchk(cudaMalloc(&d_diffConvolved, Elements(Dims*this->super) * sizeof(*d_diffConvolved)));
		cudaErrchk(cudaMemcpy(d_diffConvolved, diffConvolved.data, Elements(Dims*this->super) * sizeof(*d_diffConvolved), cudaMemcpyHostToDevice));
		if (writeDebug)
			outputDeviceAsImage(d_diffConvolved, this->Dims*this->super, tmpDir + "d_diffConvolved.mrc");
		tcomplex *d_fftDiffConvolved;
		cudaErrchk(cudaMalloc(&d_fftDiffConvolved, ElementsFFT(Dims*this->super) * sizeof(*d_fftDiffConvolved)));
		d_FFTR2C(d_diffConvolved, d_fftDiffConvolved, DimensionCount(this->Dims*this->super), this->Dims*this->super);
		if (writeDebug)
			outputDeviceAsImage(d_fftDiffConvolved, this->Dims * this->super, tmpDir + "d_fftDiffConvolved.mrc");

		MultidimArray<RDOUBLE> ctfRecon;
		this->ctfAtoms->RasterizeToVolume(ctfRecon, this->Dims, this->super, false);
		if (writeDebug)
			outputAsImage(ctfRecon, tmpDir + "ctfRecon.mrc");


		float *d_CTFRecon;
		cudaErrchk(cudaMalloc(&d_CTFRecon, Elements(Dims*this->super) * sizeof(*d_CTFRecon)));
		cudaErrchk(cudaMemcpy(d_CTFRecon, ctfRecon.data, Elements(Dims*this->super) * sizeof(*d_CTFRecon), cudaMemcpyHostToDevice));

		tcomplex *d_fftCTFRecon;
		cudaErrchk(cudaMalloc(&d_fftCTFRecon, ElementsFFT(Dims*this->super) * sizeof(*d_fftCTFRecon)));


		tfloat *d_absfftCTFRecon;
		cudaErrchk(cudaMalloc(&d_absfftCTFRecon, ElementsFFT(Dims*this->super) * sizeof(*d_absfftCTFRecon)));
		d_FFTR2C(d_CTFRecon, d_fftCTFRecon, DimensionCount(this->Dims), this->Dims*this->super);
		d_Abs(d_fftCTFRecon, d_absfftCTFRecon, ElementsFFT(Dims*this->super));
		
		if (writeDebug)
			outputDeviceAsImage(d_fftCTFRecon, this->Dims * this->super, tmpDir + "d_fftCTFRecon.mrc");
		if (writeDebug)
			outputDeviceAsImage(d_absfftCTFRecon, this->Dims * this->super, tmpDir + "d_absfftCTFRecon.mrc", true);
		
		d_MaxOp(d_absfftCTFRecon, 1e-2, d_absfftCTFRecon, ElementsFFT(Dims*this->super));
		if (writeDebug)
			outputDeviceAsImage(d_absfftCTFRecon, this->Dims * this->super, tmpDir + "d_absfftCTFRecon_max.mrc", true);
		d_ComplexDivideByVector(d_fftDiffConvolved, d_absfftCTFRecon, d_fftDiffConvolved, ElementsFFT(this->Dims*this->super), 1);
		if (writeDebug)
			outputDeviceAsImage(d_fftDiffConvolved, this->Dims * this->super, tmpDir + "d_fftDiffDivided.mrc");
		d_IFFTC2R(d_fftDiffConvolved, d_diffConvolved, DimensionCount(this->Dims), this->Dims*this->super, 1);
		if (writeDebug)
			outputDeviceAsImage(d_diffConvolved, this->Dims * this->super, tmpDir + "d_diffDivided.mrc");
		RealspaceVolumeUpdate(d_atomPositions, d_copyAtomIntensities, atoms->NAtoms, d_diffConvolved, this->Dims, this->super);

		cufftDestroy(planForward);
		cufftDestroy(planBackward);

		cudaErrchk(cudaFree(d_copyAtomIntensities));
		cudaErrchk(cudaFree(d_ctfs));
		cudaErrchk(cudaFree(d_diffConvolved));
		cudaErrchk(cudaFree(d_fftDiffConvolved));
		cudaErrchk(cudaFree(d_CTFRecon));
		cudaErrchk(cudaFree(d_fftCTFRecon));
		cudaErrchk(cudaFree(d_absfftCTFRecon));


	}
	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_atomIntensities));
	cudaErrchk(cudaFree(d_ctfAtomIntensities));
	return 0.0;
}

RDOUBLE PseudoProjector::VolumeUpdate(MultidimArray<RDOUBLE> &Volume, RDOUBLE shiftX, RDOUBLE shiftY) {

	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms->NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, atoms->AtomPositions.data(), atoms->NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms->NAtoms * sizeof(float)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms->AtomWeights.data(), atoms->NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	float * d_volumeUpdate;
	cudaErrchk(cudaMalloc((void**)&d_volumeUpdate, Elements(this->Dims) * sizeof(*d_volumeUpdate)));
	cudaErrchk(cudaMemcpy(d_volumeUpdate, Volume.data, Elements(this->Dims) * sizeof(*d_volumeUpdate), cudaMemcpyHostToDevice));

	float * d_superVolumeUpdate;
	cudaErrchk(cudaMalloc((void**)&d_superVolumeUpdate, Elements(this->Dims*this->super) * sizeof(*d_superVolumeUpdate)));

	d_Scale(d_volumeUpdate, d_superVolumeUpdate, this->Dims, this->super*this->Dims, gtom::T_INTERP_CUBIC, NULL, NULL, 1, NULL, NULL);
	cudaErrchk(cudaDeviceSynchronize());
	RealspaceVolumeUpdate(d_atomPositions, d_atomIntensities, atoms->NAtoms, d_superVolumeUpdate, this->Dims, this->super);
	cudaErrchk(cudaDeviceSynchronize());
	cudaErrchk(cudaMemcpy(atoms->AtomWeights.data(), d_atomIntensities, atoms->NAtoms * sizeof(float), cudaMemcpyDeviceToHost));

	cudaErrchk(cudaFree(d_atomPositions));
	cudaErrchk(cudaFree(d_atomIntensities));
	cudaErrchk(cudaFree(d_volumeUpdate));
	cudaErrchk(cudaFree(d_superVolumeUpdate));

	return 0;
}