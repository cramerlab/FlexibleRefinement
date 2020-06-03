
#include "PseudoProjector.h"


void PseudoProjector::project_Pseudo(DOUBLE * out, DOUBLE * out_nrm,
	float3 angles, DOUBLE shiftX, DOUBLE shiftY,
	int direction)
{
	MultidimArray<DOUBLE> proj = MultidimArray<DOUBLE>(1, this->Dims.y, this->Dims.x);
	proj.data = out;
	proj.xinit = shiftX;
	proj.yinit = shiftY;
	proj.destroyData = false;
	
	MultidimArray<DOUBLE> proj_nrm = MultidimArray<DOUBLE>(1, this->Dims.y, this->Dims.x);
	if (out_nrm != NULL)
	{
		proj_nrm.data = out_nrm;
		proj_nrm.destroyData = NULL;
	}
	proj_nrm.xinit = shiftX;
	proj_nrm.yinit = shiftY;
	
	Matrix2D<DOUBLE> EulerMat = Matrix2D<DOUBLE>();
	Euler_angles2matrix(angles.x, angles.y, angles.z, EulerMat);
	
	
	this->project_Pseudo(proj, proj_nrm,	EulerMat, shiftX, shiftY,direction);
}



void PseudoProjector::project_PseudoCTF(DOUBLE * out, DOUBLE * out_nrm, DOUBLE *gaussTable, DOUBLE * gaussTable2, DOUBLE border,
	float3 Euler, DOUBLE shiftX, DOUBLE shiftY,
	int direction)
{
	if (mode != ATOM_GAUSSIAN)
		REPORT_ERROR(std::string("This projection type is not implemented ") + __FILE__ + ": " + std::to_string(__LINE__));
	DOUBLE * tableTmp = gaussianProjectionTable.vdata;
	DOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;
	double oldBorder = this->tableLength;
	this->tableLength = border;
	gaussianProjectionTable.vdata = gaussTable;
	gaussianProjectionTable2.vdata = gaussTable2;

	MultidimArray<DOUBLE> proj = MultidimArray<DOUBLE>(1, this->Dims.y, this->Dims.x);
	proj.data = out;
	proj.xinit = shiftX;
	proj.yinit = shiftY;
	proj.destroyData = false;
	
	MultidimArray<DOUBLE> proj_nrm = MultidimArray<DOUBLE>(1, this->Dims.y, this->Dims.x);
	if (out_nrm != NULL)
	{
		proj_nrm.data = out_nrm;
		proj_nrm.destroyData = NULL;
	}
	proj_nrm.xinit = shiftX;
	proj_nrm.yinit = shiftY;
	
	Matrix2D<DOUBLE> EulerMat = Matrix2D<DOUBLE>();
	Euler_angles2matrix(Euler.x, Euler.y, Euler.z, EulerMat);
	
	
	this->project_Pseudo(proj, proj_nrm,	EulerMat, shiftX, shiftY,direction);
	gaussianProjectionTable.vdata = tableTmp;
	gaussianProjectionTable2.vdata = tableTmp2;
	this->tableLength = oldBorder;
}


MRCImage<DOUBLE> *PseudoProjector::create3DImage(DOUBLE oversampling) {
	MultidimArray<DOUBLE> data(Dims.z*oversampling, Dims.y*oversampling, Dims.x*oversampling);
	MRCImage<DOUBLE> *Volume = new MRCImage<DOUBLE>();
	atoms.RasterizeToVolume(Volume->data, Dims, oversampling);
	return Volume;
}

void PseudoProjector::project_Pseudo(MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj, std::vector<Matrix1D<DOUBLE>> * atomPositions,
	Matrix2D<DOUBLE> &Euler, DOUBLE shiftX, DOUBLE shiftY,
	int direction) {
	// Project all pseudo atoms ............................................
	int numAtoms = (*atomPositions).size();
	Matrix1D<DOUBLE> actprj(3);
	double sigma4 = this->tableLength;
	Matrix1D<DOUBLE> actualAtomPosition;

	for (int n = 0; n < numAtoms; n++)
	{
		actualAtomPosition = (*atomPositions)[n];
		XX(actualAtomPosition) -= Dims.x / 2;
		YY(actualAtomPosition) -= Dims.y / 2;
		ZZ(actualAtomPosition) -= Dims.z / 2;

		DOUBLE weight = atoms.AtomWeights[n];

		Uproject_to_plane(actualAtomPosition, Euler, actprj);
		XX(actprj) += Dims.x / 2;
		YY(actprj) += Dims.y / 2;
		ZZ(actprj) += Dims.z / 2;

		XX(actprj) += shiftX;
		YY(actprj) += shiftY;
		if (mode == ATOM_INTERPOLATE) {
			if (direction == PSEUDO_FORWARD) {
				int X0 = (int)XX(actprj);
				DOUBLE ix = XX(actprj) - X0;
				int X1 = X0 + 1;

				int Y0 = (int)YY(actprj);
				DOUBLE iy = YY(actprj) - Y0;
				int Y1 = Y0 + 1;

				DOUBLE v0 = 1.0f - iy;
				DOUBLE v1 = iy;

				DOUBLE v00 = (1.0f - ix) * v0;
				DOUBLE v10 = ix * v0;
				DOUBLE v01 = (1.0f - ix) * v1;
				DOUBLE v11 = ix * v1;

				A2D_ELEM(proj, Y0, X0) += weight * v00;
				A2D_ELEM(proj, Y0, X1) += weight * v01;
				A2D_ELEM(proj, Y1, X0) += weight * v10;
				A2D_ELEM(proj, Y1, X1) += weight * v11;
			}
			else if (direction == PSEUDO_BACKWARD) {

				int X0 = (int)XX(actprj);
				DOUBLE ix = XX(actprj) - X0;
				int X1 = X0 + 1;

				int Y0 = (int)YY(actprj);
				DOUBLE iy = YY(actprj) - Y0;
				int Y1 = Y0 + 1;

				DOUBLE v00 = A3D_ELEM(norm_proj, 0, Y0, X0);
				DOUBLE v01 = A3D_ELEM(norm_proj, 0, Y0, X1);
				DOUBLE v10 = A3D_ELEM(norm_proj, 0, Y1, X0);
				DOUBLE v11 = A3D_ELEM(norm_proj, 0, Y1, X1);


				DOUBLE v0 = Lerp(v00, v01, ix);
				DOUBLE v1 = Lerp(v10, v11, ix);

				DOUBLE v = Lerp(v0, v1, iy);

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
				DOUBLE vol_corr = 0;

				// Effectively project this basis
				for (int y = YY_corner1; y <= YY_corner2; y++)
				{
					DOUBLE y_diff2 = y - YY(actprj);
					y_diff2 = y_diff2 * y_diff2;
					for (int x = XX_corner1; x <= XX_corner2; x++)
					{
						DOUBLE x_diff2 = x - XX(actprj);
						x_diff2 = x_diff2 * x_diff2;
						DOUBLE r = sqrt(x_diff2 + y_diff2);
						DOUBLE didx = r * GAUSS_FACTOR;
						int idx = ROUND(didx);
						DOUBLE a = VEC_ELEM(gaussianProjectionTable, idx);
						DOUBLE a2 = VEC_ELEM(gaussianProjectionTable2, idx);

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
	MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj,
	Matrix2D<DOUBLE> &Euler, DOUBLE shiftX, DOUBLE shiftY,
	int direction)
{
	// Project all pseudo atoms ............................................
	int numAtoms = atoms.AtomPositions.size();
	Matrix1D<DOUBLE> actprj(3);
	double sigma4 = this->tableLength;
	Matrix1D<DOUBLE> actualAtomPosition;

	for (int n = 0; n < numAtoms; n++)
	{
		actualAtomPosition = atoms.AtomPositions[n];
		XX(actualAtomPosition) -= Dims.x / 2;
		YY(actualAtomPosition) -= Dims.y / 2;
		ZZ(actualAtomPosition) -= Dims.z / 2;

		DOUBLE weight = atoms.AtomWeights[n];

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
				DOUBLE vol_corr = 0;

				// Effectively project this basis
				for (int y = YY_corner1; y <= YY_corner2; y++)
				{
					DOUBLE y_diff2 = y - YY(actprj);
					y_diff2 = y_diff2 * y_diff2;
					for (int x = XX_corner1; x <= XX_corner2; x++)
					{
						DOUBLE x_diff2 = x - XX(actprj);
						x_diff2 = x_diff2 * x_diff2;
						DOUBLE r = sqrt(x_diff2 + y_diff2);
						DOUBLE didx = r * GAUSS_FACTOR;
						int idx = ROUND(didx);
						DOUBLE a = VEC_ELEM(gaussianProjectionTable, idx);
						DOUBLE a2 = VEC_ELEM(gaussianProjectionTable2, idx);

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
		else
			REPORT_ERROR(std::string("This projection type is not implemented ") + __FILE__ + ": " + std::to_string(__LINE__));

	}

}

DOUBLE PseudoProjector::ART_single_image(const MultidimArray<DOUBLE> &Iexp, MultidimArray<DOUBLE> &Itheo, MultidimArray<DOUBLE> &Icorr, MultidimArray<DOUBLE> &Idiff, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY)
{
	Idiff.initZeros();
	Itheo.initZeros();
	Icorr.initZeros();
	Matrix2D<DOUBLE> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_FORWARD);
	//Idiff.resize(Iexp);

	DOUBLE mean_error = 0;
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

	DOUBLE minIntensity = std::numeric_limits<DOUBLE>::max();
	DOUBLE maxIntensity = std::numeric_limits<DOUBLE>::lowest();
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
	DOUBLE a = 2.0 / (maxIntensity - minIntensity);


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
		DOUBLE intensity = atoms.AtomWeights[n];
		/*if (allowIntensity)
			intensity = a*(atomWeight[n] - minIntensity) - 1;*/
		if (col == 1)
			fprintf(fhOut,
				"ATOM  %5d DENS DENS %7d    %8.3f %8.3f %8.3f %.8f     1      DENS\n",
				n + 1, n + 1,
				(float)(atoms.AtomPositions[n](0)),
				(float)(atoms.AtomPositions[n](1)),
				(float)(atoms.AtomPositions[n](2)),
				(float)intensity);
		else
			fprintf(fhOut,
				"ATOM  %5d DENS DENS %7d    %8.3f%8.3f%8.3f     1 %.8f      DENS\n",
				n + 1, n + 1,
				(float)(atoms.AtomPositions[n](0)),
				(float)(atoms.AtomPositions[n](1)),
				(float)(atoms.AtomPositions[n](2)),
				(float)intensity);
	}
	fclose(fhOut);
}

DOUBLE PseudoProjector::ART_batched(const MultidimArray<DOUBLE> &Iexp, idxtype batchSize, float3 *angles, DOUBLE shiftX, DOUBLE shiftY)
{
	MultidimArray<DOUBLE> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Idiff.resize(Iexp, false);
	Matrix2D<DOUBLE> *EulerVec = new Matrix2D<DOUBLE>[batchSize];
	DOUBLE mean_error = 0;
#pragma omp parallel
	{
		//Initialize array for the slice view
		MultidimArray<DOUBLE> tmpItheo, tmpIcorr;
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

DOUBLE PseudoProjector::ART_batched(const MultidimArray<DOUBLE> &Iexp, MultidimArray<DOUBLE> &Itheo, MultidimArray<DOUBLE> &Icorr, MultidimArray<DOUBLE> &Idiff, idxtype batchSize, float3 *angles, DOUBLE shiftX, DOUBLE shiftY)
{
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Idiff.resize(Iexp, false);
	Matrix2D<DOUBLE> *EulerVec = new Matrix2D<DOUBLE>[batchSize];
	DOUBLE mean_error = 0;
#pragma omp parallel
	{
		//Initialize array for the slice view
		MultidimArray<DOUBLE> tmpItheo, tmpIcorr;
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

DOUBLE PseudoProjector::ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE tableLength, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<DOUBLE> > Images;
	DOUBLE itError = 0.0;
	
	for (size_t i = 0; i < numImages; i++)
	{
		DOUBLE * tableTmp = gaussianProjectionTable.vdata;
		DOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;

		gaussianProjectionTable.vdata = gaussTables + i * (GAUSS_FACTOR * Dims.x / 2 );
		gaussianProjectionTable2.vdata = gaussTables2 + i * (GAUSS_FACTOR * Dims.x / 2 );
		double oldBorder = this->tableLength;
		this->tableLength = tableLength;
		MultidimArray<DOUBLE> tmp = MultidimArray<DOUBLE>(1, Dims.y, Dims.x);
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		tmp.destroyData = false;
		itError += ART_single_image(tmp, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);

		gaussianProjectionTable.vdata = tableTmp;
		gaussianProjectionTable2.vdata = tableTmp2;
		this->tableLength = oldBorder;
	}
	return itError;
}

DOUBLE PseudoProjector::ART_multi_Image_step_DB(DOUBLE * Iexp, DOUBLE * Itheo, DOUBLE * Icorr, DOUBLE * Idiff, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE tableLength, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<DOUBLE> > Images;
	DOUBLE itError = 0.0;

	for (size_t i = 0; i < numImages; i++)
	{
		DOUBLE * tableTmp = gaussianProjectionTable.vdata;
		DOUBLE * tableTmp2 = gaussianProjectionTable2.vdata;

		gaussianProjectionTable.vdata = gaussTables + i * (GAUSS_FACTOR * Dims.x / 2);
		gaussianProjectionTable2.vdata = gaussTables2 + i * (GAUSS_FACTOR * Dims.x / 2);
		double oldBorder = this->tableLength;
		this->tableLength = tableLength;
		MultidimArray<DOUBLE> tmp = MultidimArray<DOUBLE>(1, Dims.y, Dims.x);
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		tmp.destroyData = false;

		MultidimArray<DOUBLE> tmp2 = MultidimArray<DOUBLE>(1, Dims.y, Dims.x);
		tmp2.data = Itheo + i * (Dims.x*Dims.y);
		tmp2.destroyData = false;

		MultidimArray<DOUBLE> tmp3 = MultidimArray<DOUBLE>(1, Dims.y, Dims.x);
		tmp3.data = Icorr + i * (Dims.x*Dims.y);
		tmp3.destroyData = false;

		MultidimArray<DOUBLE> tmp4 = MultidimArray<DOUBLE>(1, Dims.y, Dims.x);
		tmp4.data = Idiff + i * (Dims.x*Dims.y);
		tmp4.destroyData = false;

		itError += ART_single_image(tmp, tmp2, tmp3, tmp4, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);

		gaussianProjectionTable.vdata = tableTmp;
		gaussianProjectionTable2.vdata = tableTmp2;
		this->tableLength = oldBorder;
	}
	return itError;
}

DOUBLE PseudoProjector::ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages) {

	std::vector < MultidimArray<DOUBLE> > Images;
	DOUBLE itError = 0.0;
	MultidimArray<DOUBLE> tmp = MultidimArray<DOUBLE>(1, Dims.y, Dims.x);
	for (size_t i = 0; i < numImages; i++)
	{
		
		tmp.data = Iexp + i * (Dims.x*Dims.y);
		
		itError += ART_single_image(tmp, angles[i].x, angles[i].y, angles[i].z, shiftX, shiftY);
	}
	tmp.data = NULL;
	return itError;
}

void PseudoProjector::addToPrecalcs(std::vector< projecction> &precalc, MultidimArray<DOUBLE> &Iexp, std::vector<float3> angles, std::vector<Matrix1D<DOUBLE>> *atomPositions, DOUBLE shiftX, DOUBLE shiftY) {
	Matrix2D<DOUBLE> Euler;
	Matrix1D<DOUBLE> angle(3);
	for (int n = 0; n < angles.size(); n++)
	{
		MultidimArray<DOUBLE> *tmp = new MultidimArray<DOUBLE>();
		tmp->xdim = Iexp.xdim;
		tmp->ydim = Iexp.ydim;
		tmp->yxdim = Iexp.yxdim;
		tmp->zyxdim = Iexp.yxdim;
		tmp->nzyxdim = Iexp.yxdim;
		tmp->nzyxdimAlloc = Iexp.yxdim;
		tmp->data = &(A3D_ELEM(Iexp, n, 0, 0));

		Euler_angles2matrix(angles[n].x, angles[n].y, angles[n].z, Euler);
		XX(angle) = angles[n].x;
		YY(angle) = angles[n].y;
		ZZ(angle) = angles[n].z;
		precalc.push_back({ atomPositions, tmp, angle, Euler });
	}
}

std::vector<projecction> PseudoProjector::getPrecalcs(MultidimArray<DOUBLE> Iexp, std::vector<float3> angles, DOUBLE shiftX, DOUBLE shiftY)
{
	std::vector<projecction> precalc;
	Matrix2D<DOUBLE> Euler;
	Matrix1D<DOUBLE> angle(3);
	for (int n = 0; n < angles.size(); n++)
	{
		MultidimArray<DOUBLE> *tmp = new MultidimArray<DOUBLE>();
		tmp->xdim = Iexp.xdim;
		tmp->ydim = Iexp.ydim;
		tmp->yxdim = Iexp.yxdim;
		tmp->zyxdim = Iexp.yxdim;
		tmp->nzyxdim = Iexp.yxdim;
		tmp->nzyxdimAlloc = Iexp.yxdim;
		tmp->data = &(A3D_ELEM(Iexp, n, 0, 0));

		Euler_angles2matrix(angles[n].x, angles[n].y, angles[n].z, Euler);
		XX(angle) = angles[n].x;
		YY(angle) = angles[n].y;
		ZZ(angle) = angles[n].z;
		precalc.push_back({ &atoms.AtomPositions, tmp, angle, Euler});
	}
	return precalc;
}

DOUBLE PseudoProjector::SIRT_from_precalc(std::vector<projecction>& precalc, DOUBLE shiftX, DOUBLE shiftY)
{
	MultidimArray<DOUBLE> Itheo, Icorr, Idiff, Inorm;
	return SIRT_from_precalc(precalc, Itheo, Icorr, Idiff, Inorm, shiftX, shiftY);
}

DOUBLE PseudoProjector::SIRT_from_precalc(std::vector<projecction>& precalc, MultidimArray<DOUBLE>& Itheo, MultidimArray<DOUBLE>& Icorr, MultidimArray<DOUBLE>& Idiff, MultidimArray<DOUBLE>& Inorm, DOUBLE shiftX, DOUBLE shiftY)
{
	Itheo.initZeros(precalc.size(), super*Dims.y, super*Dims.x);
	Icorr.initZeros(precalc.size(), Dims.y, Dims.x);
	Inorm.initZeros(precalc.size(), Dims.y, Dims.x);
	Idiff.resize(precalc.size(), Dims.y, Dims.x);
	DOUBLE mean_error = 0.0;
#pragma omp parallel
	{
		//Initialize array for the slice view
		MultidimArray<DOUBLE> tmpItheo, tmpIcorr, tmpInorm;
		tmpInorm.xdim = tmpItheo.xdim = tmpIcorr.xdim = Dims.x;
		tmpInorm.ydim = tmpItheo.ydim = tmpIcorr.ydim = Dims.y;
		tmpInorm.destroyData = tmpItheo.destroyData = tmpIcorr.destroyData = false;
		tmpItheo.yxdim = tmpItheo.nzyxdim = tmpItheo.zyxdim = tmpItheo.xdim*tmpItheo.ydim;
		tmpInorm.yxdim = tmpIcorr.yxdim = tmpIcorr.nzyxdim = tmpIcorr.zyxdim = tmpIcorr.xdim*tmpItheo.ydim;

#pragma omp for
		for (int imgIdx = 0; imgIdx < precalc.size(); imgIdx++) {
			tmpItheo.data = &(A3D_ELEM(Itheo, imgIdx, 0, 0));
			tmpIcorr.data = &(A3D_ELEM(Icorr, imgIdx, 0, 0));
			tmpInorm.data = &(A3D_ELEM(Inorm, imgIdx, 0, 0));

			MultidimArray<DOUBLE> *Iexp = precalc[imgIdx].image;

			this->project_Pseudo(tmpItheo, tmpInorm, precalc[imgIdx].atomPositons,
				precalc[imgIdx].Euler, shiftX, shiftY, PSEUDO_FORWARD);


			
			FOR_ALL_DIRECT_ELEMENTS_IN_ARRAY2D((*Iexp))
			{
				// Compute difference image and error

				/*DIRECT_A3D_ELEM(Idiff, imgIdx, i, j) = DIRECT_A3D_ELEM((*Iexp), 0, i, j) - DIRECT_A3D_ELEM(Itheo, imgIdx, i, j);
				mean_error += DIRECT_A3D_ELEM(Idiff, imgIdx, i, j) * DIRECT_A3D_ELEM(Idiff, imgIdx, i, j);

				// Compute the correction image

				DIRECT_A3D_ELEM(Inorm, imgIdx, i, j) = XMIPP_MAX(DIRECT_A3D_ELEM(Inorm, imgIdx, i, j), 1);
				*/
				DIRECT_A3D_ELEM(Icorr, imgIdx, i, j) =
					this->lambdaART *(DIRECT_A3D_ELEM((*Iexp), 0, i, j) - DIRECT_A3D_ELEM(Itheo, imgIdx, i, j));
			}
			
		}
	}

		mean_error /= YXSIZE(Itheo);
#pragma omp parallel
		{
			MultidimArray<DOUBLE> tmpItheo, tmpIcorr, tmpInorm;
#pragma omp for
		for (int imgIdx = 0; imgIdx < precalc.size(); imgIdx++) {
			tmpItheo.data = &(A3D_ELEM(Itheo, imgIdx, 0, 0));
			tmpIcorr.data = &(A3D_ELEM(Icorr, imgIdx, 0, 0));

			this->project_Pseudo(tmpItheo, tmpIcorr, precalc[imgIdx].atomPositons,
				precalc[imgIdx].Euler, shiftX, shiftY, PSEUDO_BACKWARD);
		}
	}
	{
		//MRCImage<DOUBLE> tmpIm(Itheo);
		//tmpIm.writeAs<float>("D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\Itheo.mrc", true);
	}
	{
		//MRCImage<DOUBLE> tmpIm(Icorr);
		//tmpIm.writeAs<float>("D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\Icorr.mrc",true);
	}
	{
		//MRCImage<DOUBLE> tmpIm(Idiff);
		//tmpIm.writeAs<float>("D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\Idiff.mrc",true);
	}
	{
		//MRCImage<DOUBLE> tmpIm(Inorm);
		//tmpIm.writeAs<float>("D:\\EMD\\9233\\emd_9233_Scaled_1.5_75k_bs64_it15_moving\\Inorm.mrc",true);
	}
	return mean_error;
}

DOUBLE PseudoProjector::ART_single_image(const MultidimArray<DOUBLE>& Iexp, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY)
{
	MultidimArray<DOUBLE> Itheo, Icorr, Idiff;
	Itheo.initZeros(Iexp);
	Icorr.initZeros(Iexp);
	Matrix2D<DOUBLE> Euler;
	Euler_angles2matrix(rot, tilt, psi, Euler);
	this->project_Pseudo(Itheo, Icorr,
		Euler, shiftX, shiftY, PSEUDO_FORWARD);
	Idiff.resize(Iexp);

	DOUBLE mean_error = 0;
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

DOUBLE PseudoProjector::ART_multi_Image_step(std::vector< MultidimArray<DOUBLE> > Iexp, std::vector<float3> angles, DOUBLE shiftX, DOUBLE shiftY) {
	DOUBLE itError = 0.0;
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