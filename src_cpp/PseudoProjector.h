#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include "readMRC.h"
#include "Types.h"
#include "macros.h"
#include "funcs.h"
#include "liblionImports.h"


#define GAUSS_FACTOR 30

using namespace relion;

#define PSEUDO_FORWARD   1
#define PSEUDO_BACKWARD -1



class PseudoProjector {

public:
	// Gaussian projection table
	Matrix1D<DOUBLE> gaussianProjectionTable;

	// Gaussian projection2 table
	Matrix1D<DOUBLE> gaussianProjectionTable2;


	std::vector< Matrix1D<DOUBLE> > atomPosition;

	// Atomic weights
	std::vector< DOUBLE > atomWeight;
	double sigma;
	double tableLength;
	double lambdaART;
	int3 Dims;
	bool gauss2D;



	PseudoProjector(int3 dims, DOUBLE *atomPositions, DOUBLE *atomWeights, DOUBLE sigma, unsigned int nAtoms):Dims(dims), sigma(sigma), lambdaART(0.1){
		atomPosition = std::vector<Matrix1D<DOUBLE>>();
		atomPosition.reserve(nAtoms);

		atomWeight.reserve(nAtoms);
		for (size_t i = 0; i < nAtoms; i++)
		{
			Matrix1D<DOUBLE> tmp = Matrix1D<DOUBLE>(3);
			XX(tmp) = atomPositions[i * 3];
			YY(tmp) = atomPositions[i * 3 + 1];
			ZZ(tmp) = atomPositions[i * 3 + 2];
			atomPosition.push_back(tmp);
			atomWeight.push_back(atomWeights[i]);
		}
		
		
		
		DOUBLE sigma4 = 4 * sigma;
		tableLength = sigma4;
		gaussianProjectionTable = Matrix1D<DOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		gaussianProjectionTable2 = Matrix1D<DOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
			gaussianProjectionTable(i) = this->gaussian1D(i / ((DOUBLE)GAUSS_FACTOR), sigma);
		gaussianProjectionTable *= gaussian1D(0, sigma);
		gaussianProjectionTable2 = gaussianProjectionTable;
		gaussianProjectionTable2 *= gaussianProjectionTable;
	};

	DOUBLE gaussian1D(DOUBLE x, DOUBLE sigma, DOUBLE mu=0)
	{
		x -= mu;
		return exp(-0.5*((x / sigma)*(x / sigma)));
	}

	DOUBLE ART_single_image(const MultidimArray<DOUBLE> &Iexp, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE ART_single_image(const MultidimArray<DOUBLE> &Iexp, MultidimArray<DOUBLE> &Itheo, MultidimArray<DOUBLE> &Icorr, MultidimArray<DOUBLE> &Idiff, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY);

	DOUBLE ART_batched(const MultidimArray<DOUBLE> &Iexp, idxtype batchSize, float3 *angles, DOUBLE shiftX, DOUBLE shiftY);

	DOUBLE ART_multi_Image_step(std::vector< MultidimArray<DOUBLE> > Iexp, std::vector<float3> angles, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE border, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
	DOUBLE ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
	DOUBLE ART_multi_Image_step_DB(DOUBLE * Iexp, DOUBLE * Itheo, DOUBLE * Icorr, DOUBLE * Idiff, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE tableLength, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

	MRCImage<DOUBLE> create3DImage(DOUBLE oversampling = 1.0);

	void project_Pseudo_batch(MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj,
		std::vector<Matrix2D<DOUBLE>> &EulerVec, DOUBLE shiftX, DOUBLE shiftY, int direction);

	void project_Pseudo(MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj,
		Matrix2D<DOUBLE> &Euler, DOUBLE shiftX, DOUBLE shiftY,
		int direction);
	void project_PseudoCTF(DOUBLE * out, DOUBLE * out_nrm, DOUBLE *gaussTable, DOUBLE * gaussTable2, DOUBLE border,
		float3 Euler, DOUBLE shiftX, DOUBLE shiftY,
		int direction);
	void project_Pseudo(DOUBLE * out,
						DOUBLE * out_nrm,
		                float3 angles,
						DOUBLE shiftX,
						DOUBLE shiftY,
		                int direction);

	void writePDB(FileName outpath);

};



