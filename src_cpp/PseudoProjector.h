#pragma once
#ifndef  PSEUDO_PROJECTOR
#define PSEUDO_PROJECTOR
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include "liblionImports.h"
#include "Types.h"
#include "readMRC.h"
#include "macros.h"
#include "funcs.h"
#include "pseudoatoms.h"

#define GAUSS_FACTOR 30

using namespace relion;

#define PSEUDO_FORWARD   1
#define PSEUDO_BACKWARD -1


struct projecction {
	std::vector< Matrix1D<DOUBLE> > *atomPositons;
	MultidimArray<DOUBLE> *image;
	Matrix1D<DOUBLE> angles;
	Matrix2D<DOUBLE> Euler;
};

class PseudoProjector {

public:
	// Gaussian projection table
	Matrix1D<DOUBLE> gaussianProjectionTable;

	// Gaussian projection2 table
	Matrix1D<DOUBLE> gaussianProjectionTable2;

	PseudoAtomMode mode;

	Pseudoatoms atoms;

	//std::vector< Matrix1D<DOUBLE> > atomPositions;
	
	// Atomic weights
	//std::vector< DOUBLE > atomWeight;
	double sigma;
	double super;
	double tableLength;
	double lambdaART;
	int3 Dims;
	bool gauss2D;


	//PseudoProjector with Gaussian mode
	PseudoProjector(int3 dims, DOUBLE *atomPositionCArr, DOUBLE *atomWeights, DOUBLE sigma, DOUBLE super, unsigned int nAtoms):Dims(dims), sigma(sigma), super(super),mode(ATOM_GAUSSIAN), atoms(atomPositionCArr, atomWeights, nAtoms, ATOM_GAUSSIAN, sigma, GAUSS_FACTOR), lambdaART(0.1){	
		
		DOUBLE sigma4 = 4 * sigma;
		tableLength = sigma4;
		gaussianProjectionTable = Matrix1D<DOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		gaussianProjectionTable2 = Matrix1D<DOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
			gaussianProjectionTable(i) = gaussian1D(i / ((DOUBLE)GAUSS_FACTOR), sigma);
		gaussianProjectionTable *= gaussian1D(0, sigma);
		gaussianProjectionTable2 = gaussianProjectionTable;
		gaussianProjectionTable2 *= gaussianProjectionTable;
	};

	PseudoProjector(int3 dims, DOUBLE *atomPositionCArr, DOUBLE *atomWeights, DOUBLE super, unsigned int nAtoms) :Dims(dims), sigma(0), super(super),mode(ATOM_INTERPOLATE), atoms(atomPositionCArr, atomWeights, nAtoms, ATOM_INTERPOLATE), lambdaART(0.1) {
	};



	std::vector<projecction> getPrecalcs(MultidimArray<DOUBLE> Iexp, std::vector<float3> angles, DOUBLE shiftX, DOUBLE shiftY);
	void addToPrecalcs(std::vector< projecction> &precalc, MultidimArray<DOUBLE> &Iexp, std::vector<float3> angles, std::vector<Matrix1D<DOUBLE>> *atomPositions, DOUBLE shiftX, DOUBLE shiftY);

	DOUBLE SIRT_from_precalc(MultidimArray<DOUBLE> &Iexp, std::vector<projecction> &precalc, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE SIRT_from_precalc(MultidimArray<DOUBLE> &Iexp, std::vector<projecction>& precalc, MultidimArray<DOUBLE>& Itheo, MultidimArray<DOUBLE>& Icorr, MultidimArray<DOUBLE>& Idiff, MultidimArray<DOUBLE>& Inorm, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE ART_single_image(const MultidimArray<DOUBLE> &Iexp, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE ART_single_image(const MultidimArray<DOUBLE> &Iexp, MultidimArray<DOUBLE> &Itheo, MultidimArray<DOUBLE> &Icorr, MultidimArray<DOUBLE> &Idiff, DOUBLE rot, DOUBLE tilt, DOUBLE psi, DOUBLE shiftX, DOUBLE shiftY);

	DOUBLE ART_batched(const MultidimArray<DOUBLE> &Iexp, idxtype batchSize, float3 *angles, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE ART_batched(const MultidimArray<DOUBLE> &Iexp, MultidimArray<DOUBLE> &Itheo, MultidimArray<DOUBLE> &Icorr, MultidimArray<DOUBLE> &Idiff, idxtype batchSize, float3 *angles, DOUBLE shiftX, DOUBLE shiftY);

	DOUBLE ART_multi_Image_step(std::vector< MultidimArray<DOUBLE> > Iexp, std::vector<float3> angles, DOUBLE shiftX, DOUBLE shiftY);
	DOUBLE ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE border, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
	DOUBLE ART_multi_Image_step(DOUBLE * Iexp, float3 * angles, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);
	DOUBLE ART_multi_Image_step_DB(DOUBLE * Iexp, DOUBLE * Itheo, DOUBLE * Icorr, DOUBLE * Idiff, float3 * angles, DOUBLE *gaussTables, DOUBLE *gaussTables2, DOUBLE tableLength, DOUBLE shiftX, DOUBLE shiftY, unsigned int numImages);

	MRCImage<DOUBLE> *create3DImage(DOUBLE oversampling = 1.0);

	void project_Pseudo_batch(MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj,
		std::vector<Matrix2D<DOUBLE>> &EulerVec, DOUBLE shiftX, DOUBLE shiftY, int direction);

	void project_Pseudo(MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj,
		Matrix2D<DOUBLE> &Euler, DOUBLE shiftX, DOUBLE shiftY,
		int direction);

	void project_Pseudo(MultidimArray<DOUBLE> &proj, MultidimArray<DOUBLE> &norm_proj, std::vector<Matrix1D<DOUBLE>> * atomPositions,
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

	void setAtomPositions(std::vector<Matrix1D<DOUBLE>> newPositions) {
		atoms.AtomPositions = newPositions;
	}

	void writePDB(FileName outpath);

};

#endif // ! PSEUDO_PROJECTOR

