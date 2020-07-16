
#include "PseudoProjector.h"
#include "gpu_project.cuh"
#include "cudaHelpers.cuh"
void PseudoProjector::project_Pseudo(RDOUBLE * out, RDOUBLE * out_nrm,
	float3 angles, RDOUBLE shiftX, RDOUBLE shiftY,
	int direction)
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
	
	
	this->project_Pseudo(proj, proj_nrm,	EulerMat, shiftX, shiftY,direction);
}



void PseudoProjector::project_PseudoCTF(RDOUBLE * out, RDOUBLE * out_nrm, RDOUBLE *gaussTable, RDOUBLE * gaussTable2, RDOUBLE border,
	float3 Euler, RDOUBLE shiftX, RDOUBLE shiftY,
	int direction)
{
	if (mode != ATOM_GAUSSIAN)
		REPORT_ERROR(std::string("This projection type is not implemented ") + __FILE__ + ": " + std::to_string(__LINE__));
	RDOUBLE * tableTmp = gaussianProjectionTable.vdata;
	RDOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;
	double oldBorder = this->tableLength;
	this->tableLength = border;
	gaussianProjectionTable.vdata = gaussTable;
	gaussianProjectionTable2.vdata = gaussTable2;

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
	Euler_angles2matrix(Euler.x, Euler.y, Euler.z, EulerMat);
	
	
	this->project_Pseudo(proj, proj_nrm,	EulerMat, shiftX, shiftY,direction);
	gaussianProjectionTable.vdata = tableTmp;
	gaussianProjectionTable2.vdata = tableTmp2;
	this->tableLength = oldBorder;
}


MRCImage<RDOUBLE> *PseudoProjector::create3DImage(RDOUBLE oversampling) {

	MRCImage<RDOUBLE> *Volume = new MRCImage<RDOUBLE>();
	atoms.RasterizeToVolume(Volume->data, Dims, oversampling);
	Volume->header.dimensions = Dims;
	return Volume;
}

void PseudoProjector::project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj, std::vector<Matrix1D<RDOUBLE>> * atomPositions,
	Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY,
	int direction) {
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

		RDOUBLE weight = atoms.AtomWeights[n];

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
					atoms.AtomWeights[n] += vol_corr;
					//atomWeight[n] = atomWeight[n] > 0 ? atomWeight[n] : 0;
				}
			}

		} // If not collapsed
		
	}
}

 /** Projection of a pseudoatom volume */
void PseudoProjector::project_Pseudo(
	MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj,
	Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY,
	int direction)
{
	// Project all pseudo atoms ............................................
	int numAtoms = atoms.AtomPositions.size();
	Matrix1D<RDOUBLE> actprj(3);
	double sigma4 = this->tableLength;
	Matrix1D<RDOUBLE> actualAtomPosition;

	for (int n = 0; n < numAtoms; n++)
	{

		XX(actualAtomPosition) = atoms.AtomPositions[n].x - Dims.x / 2;
		YY(actualAtomPosition) = atoms.AtomPositions[n].y - Dims.y / 2;
		ZZ(actualAtomPosition) = atoms.AtomPositions[n].z - Dims.z / 2;

		RDOUBLE weight = atoms.AtomWeights[n];

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
					atoms.AtomWeights[n] += vol_corr;
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

void PseudoProjector::writePDB(FileName outPath) {

	RDOUBLE minIntensity = std::numeric_limits<RDOUBLE>::max();
	RDOUBLE maxIntensity = std::numeric_limits<RDOUBLE>::lowest();
	// Write the PDB
	for each (auto var in atoms.AtomWeights)
	{
		minIntensity = std::min(minIntensity, var);
		maxIntensity = std::max(maxIntensity, var);
	}
	bool allowIntensity = true;
	if (maxIntensity - minIntensity < 1e-4)
	{

		allowIntensity = false;
	}
	RDOUBLE a = 2.0 / (maxIntensity - minIntensity);


	FILE *fhOut = NULL;
	fhOut = fopen((outPath + ".pdb").c_str(), "w");
	if (!fhOut)
		REPORT_ERROR(outPath + ".pdb");
	idxtype nmax = atoms.AtomWeights.size();
	idxtype col = 1;

	fprintf(fhOut, "REMARK pseudo_projector\n");
	fprintf(fhOut, "REMARK fixedGaussian %lf\n", sigma);
	/*fprintf(fhOut, "REMARK Scaled ints %lf\n", a);
	fprintf(fhOut, "REMARK min int %lf\n", minIntensity);*/
	fprintf(fhOut, "REMARK intensityColumn Bfactor\n");
	for (idxtype n = 0; n < nmax; n++)
	{
		RDOUBLE intensity = atoms.AtomWeights[n];
		/*if (allowIntensity)
			intensity = a*(atomWeight[n] - minIntensity) - 1;*/
		if (col == 1)
			fprintf(fhOut,
				"ATOM  %5d DENS DENS %7d    %8.3f %8.3f %8.3f %.8f     1      DENS\n",
				n + 1, n + 1,
				(float)(atoms.AtomPositions[n].x),
				(float)(atoms.AtomPositions[n].y),
				(float)(atoms.AtomPositions[n].z),
				(float)intensity);
		else
			fprintf(fhOut,
				"ATOM  %5d DENS DENS %7d    %8.3f%8.3f%8.3f     1 %.8f      DENS\n",
				n + 1, n + 1,
				(float)(atoms.AtomPositions[n].x),
				(float)(atoms.AtomPositions[n].y),
				(float)(atoms.AtomPositions[n].z),
				(float)intensity);
	}
	fclose(fhOut);
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

		gaussianProjectionTable.vdata = gaussTables + i * (GAUSS_FACTOR * Dims.x / 2 );
		gaussianProjectionTable2.vdata = gaussTables2 + i * (GAUSS_FACTOR * Dims.x / 2 );
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

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, RDOUBLE shiftX, RDOUBLE shiftY)
{

	return SIRT(Iexp, angles, numAngles, NULL, NULL, NULL, NULL, shiftX, shiftY);
}

void PseudoProjector::projectForward(float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>& Itheo, RDOUBLE shiftX, RDOUBLE shiftY)
{
	cudaErrchk(cudaSetDevice(1));
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms.NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, atoms.AtomPositions.data(), atoms.NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms.NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms.AtomWeights.data(), atoms.NAtoms * sizeof(float), cudaMemcpyHostToDevice));

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
	Itheo.resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);


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
		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj) * sizeof(float)));

		float * d_projections;
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));


		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm += ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);


			float3 * h_angles = angles + startIm;
			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, atoms.NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			cudaErrchk(cudaMemcpy(Itheo.data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
		}
		cudaFree(d_superProjections);
		cudaFree(d_projections);
	}
	cudaFree(d_atomIntensities);
	cudaFree(d_atomPositions);
}

RDOUBLE PseudoProjector::SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* superICorr, RDOUBLE shiftX, RDOUBLE shiftY)
{
	cudaErrchk(cudaSetDevice(1));
	float3 * d_atomPositions;
	cudaErrchk(cudaMalloc((void**)&d_atomPositions, atoms.NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomPositions, atoms.AtomPositions.data(), atoms.NAtoms * sizeof(float3), cudaMemcpyHostToDevice));

	float * d_atomIntensities;
	cudaErrchk(cudaMalloc((void**)&d_atomIntensities, atoms.NAtoms * sizeof(float3)));
	cudaErrchk(cudaMemcpy(d_atomIntensities, atoms.AtomWeights.data(), atoms.NAtoms * sizeof(float), cudaMemcpyHostToDevice));

	idxtype GPU_FREEMEM;
	idxtype GPU_MEMLIMIT;
	cudaMemGetInfo(&GPU_FREEMEM, &GPU_MEMLIMIT);

	int3 dimsvolume = {Dims.x, Dims.y, Dims.z };
	int3 superDimsvolume = { Dims.x * super, Dims.y * super, Dims.z * super };

	int2 superDimsproj = { Dims.x * super, Dims.y * super };
	int2 dimsproj = { Dims.x,  Dims.y };

	idxtype space = Elements2(dimsproj) * sizeof(float);
	idxtype ElementsPerBatch = 0.9*(GPU_FREEMEM / ((2*Elements2(superDimsproj)+ Elements2(dimsproj)+ Elements2(dimsproj)) * sizeof(float)));

	ElementsPerBatch = std::min(numAngles, (idxtype)2048);	//Hard limit of elementsPerBatch instead of calculating
	Itheo->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	Icorr->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
	Idiff->resizeNoCopy(numAngles, dimsproj.y, dimsproj.x);
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
		cudaErrchk(cudaMalloc(&d_superProjections, ElementsPerBatch*Elements2(superDimsproj)*sizeof(float)));
		float * d_iExp;
		cudaErrchk(cudaMalloc(&d_iExp,ElementsPerBatch*Elements2(dimsproj) * sizeof(float)));
		float * d_projections;
		cudaErrchk(cudaMalloc(&d_projections, numBatches*ElementsPerBatch * Elements2(dimsproj) * sizeof(float)));
		

		// Forward project in batches
		for (idxtype startIm = 0; startIm < numAngles; startIm+=ElementsPerBatch)
		{
			idxtype batch = std::min(ElementsPerBatch, numAngles - startIm);
			float * d_projectionsBatch = d_projections + startIm * Elements2(dimsproj);


			float3 * h_angles = angles+startIm;
			cudaMemcpy(d_iExp, Iexp.data+startIm*Elements2(dimsproj), batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyHostToDevice);
			RealspacePseudoProjectForward(d_atomPositions, d_atomIntensities, atoms.NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());

			//We have planned for ElementsPerBatch many transforms, therefore we scale also the non existing parts between the end of batch and the end of d_superProj
			d_Scale(d_superProjections, d_projectionsBatch, gtom::toInt3(superDimsproj), gtom::toInt3(dimsproj), T_INTERP_FOURIER, &planForward, &planBackward, ElementsPerBatch);
			cudaErrchk(cudaMemcpy(Itheo->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_SubtractVector(d_projectionsBatch, d_iExp, d_projectionsBatch, batch*Elements2(dimsproj), 1);
			cudaErrchk(cudaMemcpy(Idiff->data + startIm * Elements2(dimsproj), d_projectionsBatch, batch*Elements2(dimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			gtom::d_MultiplyByScalar(d_projectionsBatch, d_projectionsBatch, batch*Elements2(dimsproj), -lambdaART);
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
			cudaErrchk(cudaMemcpy(superICorr->data + startIm * Elements2(superDimsproj), d_superProjections, batch*Elements2(superDimsproj) * sizeof(float), cudaMemcpyDeviceToHost));
			RealspacePseudoProjectBackward(d_atomPositions, d_atomIntensities, atoms.NAtoms, superDimsvolume, d_superProjections, superDimsproj, super, h_angles, batch);
			cudaErrchk(cudaPeekAtLastError());
			
		}
		cudaErrchk(cudaMemcpy(atoms.AtomWeights.data(), d_atomIntensities, atoms.NAtoms*sizeof(*(atoms.AtomWeights.data())), cudaMemcpyDeviceToHost));
		cufftDestroy(planForward);
		cufftDestroy(planBackward);
		cudaFree(d_superProjections);
		cudaFree(d_projections);

	}
	cudaFree(d_atomPositions);
	cudaFree(d_atomIntensities);
	return 0.0;
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
#ifdef DEBUG
		auto Iexp_i_j = DIRECT_A2D_ELEM(Iexp, i, j);
		auto Itheo_i_j = DIRECT_A2D_ELEM(Itheo, i, j);
#endif // DEBUG



		DIRECT_A2D_ELEM(Idiff, i, j) = DIRECT_A2D_ELEM(Iexp, i, j) - DIRECT_A2D_ELEM(Itheo, i, j);
		mean_error += DIRECT_A2D_ELEM(Idiff, i, j) * DIRECT_A2D_ELEM(Idiff, i, j);

		// Compute the correction image
#ifdef DEBUG
		auto Icorr_i_j_a = DIRECT_A2D_ELEM(Icorr, i, j);
#endif DEBUG
		DIRECT_A2D_ELEM(Icorr, i, j) = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
#ifdef DEBUG
		auto Icorr_i_j_b = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
		auto Icorr_i_j_c = this->lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
#endif DEBUG
		DIRECT_A2D_ELEM(Icorr, i, j) =
			this->lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
	}
	mean_error /= YXSIZE(Iexp);

	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_BACKWARD);
	return mean_error;
}

RDOUBLE PseudoProjector::ART_multi_Image_step(std::vector< MultidimArray<RDOUBLE> > Iexp, std::vector<float3> angles, RDOUBLE shiftX, RDOUBLE shiftY) {
	RDOUBLE itError = 0.0;
	for(unsigned int i=0; i < Iexp.size(); i++)
	{
		itError += ART_single_image(Iexp[i], angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);
	}
	return itError / Iexp.size(); // Mean Error
}


/*
void ProgARTPseudo::show() const
{
	if (verbose > 0)
	{
		std::cout << " =======================================================" << std::endl;
		std::cout << " ART reconstruction method using pseudo atomic structure" << std::endl;
		std::cout << " =======================================================" << std::endl;
		std::cout << "Input images:    " << fnDoc << std::endl;
		std::cout << "Pseudoatoms:     " << fnPseudo << std::endl;
		std::cout << "Sigma:           " << sigma << std::endl;
		std::cout << "Sampling rate:   " << sampling << std::endl;
		if (!fnNMA.empty())
			std::cout << "NMA:             " << fnNMA << std::endl;
		std::cout << "Output rootname: " << fnRoot << std::endl;
		std::cout << "Lambda ART:      " << lambdaART << std::endl;
		std::cout << "N. Iterations:   " << Nit << std::endl;
		std::cout << "\n -----------------------------------------------------" << std::endl;
	}
}

void ProgARTPseudo::defineParams()
{
	addUsageLine("Generate 3D reconstructions from projections using ART on pseudoatoms.");
	addUsageLine("+This program reconstructs based on irregular grids given by pseudo atomic structures. ");
	addUsageLine("+In case of regular grids, please refer to other programs as reconstruct_art ");
	addUsageLine("+or reconstruct_fourier. Optionally, a deformation file generated by Nodal Mode ");
	addUsageLine("+Alignment (NMA) can be passed with --nma parameter.");

	addSeeAlsoLine("volume_to_pseudoatoms, nma_alignment");

	addParamsLine("   -i <md_file>          : Metadata file with input projections");
	addParamsLine("   --oroot <rootname>    : Output rootname");
	addParamsLine("   alias -o;");
	addParamsLine("   --pseudo <pseudofile> : Pseudo atomic structure (PDB format)");
	addParamsLine("     alias -p;");
	addParamsLine("  [--sigma <s=-1>]       : Pseudoatom sigma. By default, from pseudo file");
	addParamsLine("  [--sampling_rate <Ts=1>]  : Pixel size (Angstrom)");
	addParamsLine("  alias -s;");
	addParamsLine("  [-l <lambda=0.1>]      : Relaxation factor");
	addParamsLine("  [-n <N=1>]             : Number of iterations");
	addParamsLine("  [--nma <selfile=\"\">] : Selfile with NMA");

	addExampleLine("Reconstruct with NMA file and relaxation factor of 0.2:", false);
	addExampleLine("xmipp_reconstruct_art_pseudo -i projections.xmd -o art_rec --nma nmafile.xmd -l 0.2");
}

void ProgARTPseudo::readParams()
{
	fnDoc = getParam("-i");
	fnPseudo = getParam("--pseudo");
	fnRoot = getParam("--oroot");
	lambdaART = getDoubleParam("-l");
	Nit = getIntParam("-n");
	sigma = getDoubleParam("--sigma");
	fnNMA = getParam("--nma");
	sampling = getDoubleParam("--sampling_rate");
}

void ProgARTPseudo::produceSideInfo()
{
	DF.read(fnDoc);
	std::ifstream fhPseudo;
	fhPseudo.open(fnPseudo.c_str());
	if (!fhPseudo)
		REPORT_ERROR(ERR_IO_NOTEXIST, fnPseudo);
	while (!fhPseudo.eof())
	{
		std::string line;
		getline(fhPseudo, line);
		if (line.length() == 0)
			continue;
		if (line.substr(7, 13) == "fixedGaussian" && sigma < 0)
		{
			std::vector < std::string> results;
			splitString(line, " ", results);
			sigma = textToFloat(results[2]);
			sigma /= sampling;
		}
		else if (line.substr(0, 4) == "ATOM")
		{
			Matrix1D<double> v(3);
			v(0) = textToFloat(line.substr(30, 8));
			v(1) = textToFloat(line.substr(38, 8));
			v(2) = textToFloat(line.substr(46, 8));
			v /= sampling;
			atomPositions.push_back(v);
			atomWeight.push_back(0);
		}
	}
	fhPseudo.close();

	double sigma4 = 4 * sigma;
	gaussianProjectionTable.resize(CEIL(sigma4*sqrt(2) * 1000));
	FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
		gaussianProjectionTable(i) = gaussian1D(i / 1000.0, sigma);
	gaussianProjectionTable *= gaussian1D(0, sigma);
	gaussianProjectionTable2 = gaussianProjectionTable;
	gaussianProjectionTable2 *= gaussianProjectionTable;

	// NMA
	if (!fnNMA.empty())
	{
		MetaData DFNMA(fnNMA);
		DFNMA.removeDisabled();
		FOR_ALL_OBJECTS_IN_METADATA(DFNMA)
		{
			Matrix2D<double> mode;
			mode.initZeros(atomPositions.size(), 3);
			FileName fnMode;
			DFNMA.getValue(MDL_NMA_MODEFILE, fnMode, __iter.objId);
			mode.read(fnMode);
			NMA.push_back(mode);
		}
	}
}

void ProgARTPseudo::run()
{
	show();
	produceSideInfo();
	Image<double> Iexp;
	for (int it = 0; it < Nit; it++)
	{
		double itError = 0;
		FOR_ALL_OBJECTS_IN_METADATA(DF)
		{
			FileName fnExp;
			DF.getValue(MDL_IMAGE, fnExp, __iter.objId);
			double rot;
			DF.getValue(MDL_ANGLE_ROT, rot, __iter.objId);
			double tilt;
			DF.getValue(MDL_ANGLE_TILT, tilt, __iter.objId);
			double psi;
			DF.getValue(MDL_ANGLE_PSI, psi, __iter.objId);
			double shiftX;
			DF.getValue(MDL_SHIFT_X, shiftX, __iter.objId);
			double shiftY;
			DF.getValue(MDL_SHIFT_Y, shiftY, __iter.objId);
			std::vector<double> lambda;
			if (NMA.size() > 0)
				DF.getValue(MDL_NMA, lambda, __iter.objId);

			Iexp.read(fnExp);
			Iexp().setXmippOrigin();
			itError += ART_single_step(Iexp(), rot, tilt, psi, -shiftX, -shiftY, lambda);
		}
		if (DF.size() > 0)
			itError /= DF.size();
		std::cerr << "Error at iteration " << it << " = " << itError << std::endl;
	}
	writePseudo();
}

void ProgARTPseudo::writePseudo()
{
	// Convert from pseudoatoms to volume
	Image<double> V;
	size_t objId = DF.firstObject();
	FileName fnExp;
	DF.getValue(MDL_IMAGE, fnExp, objId);
	Image<double> I;
	I.read(fnExp, HEADER);
	V().resize(XSIZE(I()), XSIZE(I()), XSIZE(I()));
	V().setXmippOrigin();

	int nmax = atomPositions.size();
	double sigma4 = 4 * sigma;
	for (int n = 0; n < nmax; n++)
	{
		int XX_corner1 = CEIL(XMIPP_MAX(STARTINGX(V()), XX(atomPositions[n]) - sigma4));
		int YY_corner1 = CEIL(XMIPP_MAX(STARTINGY(V()), YY(atomPositions[n]) - sigma4));
		int ZZ_corner1 = CEIL(XMIPP_MAX(STARTINGY(V()), ZZ(atomPositions[n]) - sigma4));
		int XX_corner2 = FLOOR(XMIPP_MIN(FINISHINGX(V()), XX(atomPositions[n]) + sigma4));
		int YY_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(V()), YY(atomPositions[n]) + sigma4));
		int ZZ_corner2 = FLOOR(XMIPP_MIN(FINISHINGY(V()), ZZ(atomPositions[n]) + sigma4));
		if (XX_corner1 <= XX_corner2 && YY_corner1 <= YY_corner2 &&
			ZZ_corner1 <= ZZ_corner2)
		{
			for (int z = ZZ_corner1; z <= ZZ_corner2; z++)
				for (int y = YY_corner1; y <= YY_corner2; y++)
					for (int x = XX_corner1; x <= XX_corner2; x++)
						V(z, y, x) += atomWeight[n] *
						gaussian1D(z - ZZ(atomPositions[n]), sigma)*
						gaussian1D(y - YY(atomPositions[n]), sigma)*
						gaussian1D(x - XX(atomPositions[n]), sigma);
		}
	}
	V.write(fnRoot + ".vol");

	// Histogram of the intensities
	MultidimArray<double> intensities(atomWeight);
	Histogram1D hist;
	compute_hist(intensities, hist, 100);
	hist.write(fnRoot + "_intensities.hist");
}

double ProgARTPseudo::ART_single_step(const MultidimArray<double> &Iexp,
	double rot, double tilt, double psi, double shiftX, double shiftY,
	const std::vector<double> &lambda)
{
	MultidimArray<double> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Matrix2D<double> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	project_Pseudo(atomPositions, atomWeight, sigma, Itheo, Icorr,
		Euler, shiftX, shiftY, lambda, NMA, FORWARD,
		gaussianProjectionTable, gaussianProjectionTable2);
	Idiff.initZeros(Iexp);

	double mean_error = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D(Iexp)
	{
		// Compute difference image and error
		DIRECT_A2D_ELEM(Idiff, i, j) = DIRECT_A2D_ELEM(Iexp, i, j) - DIRECT_A2D_ELEM(Itheo, i, j);
		mean_error += DIRECT_A2D_ELEM(Idiff, i, j) * DIRECT_A2D_ELEM(Idiff, i, j);

		// Compute the correction image
		DIRECT_A2D_ELEM(Icorr, i, j) = XMIPP_MAX(DIRECT_A2D_ELEM(Icorr, i, j), 1);
		DIRECT_A2D_ELEM(Icorr, i, j) =
			lambdaART * DIRECT_A2D_ELEM(Idiff, i, j) / DIRECT_A2D_ELEM(Icorr, i, j);
	}
	mean_error /= YXSIZE(Iexp);

	project_Pseudo(atomPositions, atomWeight, sigma, Itheo, Icorr,
		Euler, shiftX, shiftY, lambda, NMA, BACKWARD,
		gaussianProjectionTable, gaussianProjectionTable2);
	return mean_error;
}*/