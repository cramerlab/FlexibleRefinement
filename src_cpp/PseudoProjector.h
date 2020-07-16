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
#include "Prerequisites.cuh"

#define GAUSS_FACTOR 30

using namespace relion;

#define PSEUDO_FORWARD   1
#define PSEUDO_BACKWARD -1


struct projecction {
	std::vector< Matrix1D<RDOUBLE> > *atomPositons;
	MultidimArray<RDOUBLE> *image;
	Matrix1D<RDOUBLE> angles;
	Matrix2D<RDOUBLE> Euler;
};

class PseudoProjector {

public:
	// Gaussian projection table
	Matrix1D<RDOUBLE> gaussianProjectionTable;

	// Gaussian projection2 table
	Matrix1D<RDOUBLE> gaussianProjectionTable2;

	PseudoAtomMode mode;

	Pseudoatoms atoms;

	//std::vector< Matrix1D<RDOUBLE> > atomPositions;
	
	// Atomic weights
	//std::vector< RDOUBLE > atomWeight;
	double sigma;
	double super;
	double tableLength;
	RDOUBLE lambdaART;
	int3 Dims;
	bool gauss2D;


	//PseudoProjector with Gaussian mode
	PseudoProjector(int3 dims, RDOUBLE *atomPositionCArr, RDOUBLE *atomWeights, RDOUBLE sigma, RDOUBLE super, unsigned int nAtoms):Dims(dims), sigma(sigma), super(super),mode(ATOM_GAUSSIAN), atoms(atomPositionCArr, atomWeights, nAtoms, ATOM_GAUSSIAN, sigma, GAUSS_FACTOR), lambdaART(0.1){	
		
		RDOUBLE sigma4 = 4 * sigma;
		tableLength = sigma4;
		gaussianProjectionTable = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		gaussianProjectionTable2 = Matrix1D<RDOUBLE>(CEIL(sigma4*sqrt(3) * GAUSS_FACTOR + 1));
		FOR_ALL_ELEMENTS_IN_MATRIX1D(gaussianProjectionTable)
			gaussianProjectionTable(i) = gaussian1D(i / ((RDOUBLE)GAUSS_FACTOR), sigma);
		gaussianProjectionTable *= gaussian1D(0, sigma);
		gaussianProjectionTable2 = gaussianProjectionTable;
		gaussianProjectionTable2 *= gaussianProjectionTable;
	};

	PseudoProjector(int3 dims, RDOUBLE *atomPositionCArr, RDOUBLE *atomWeights, RDOUBLE super, unsigned int nAtoms) :Dims(dims), sigma(0), super(super),mode(ATOM_INTERPOLATE), atoms(atomPositionCArr, atomWeights, nAtoms, ATOM_INTERPOLATE), lambdaART(0.1) {
	};



	std::vector<projecction> getPrecalcs(MultidimArray<RDOUBLE> Iexp, std::vector<float3> angles, RDOUBLE shiftX, RDOUBLE shiftY);
	void addToPrecalcs(std::vector< projecction> &precalc, MultidimArray<RDOUBLE> &Iexp, std::vector<float3> angles, std::vector<Matrix1D<RDOUBLE>> *atomPositions, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype batch, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE SIRT(MultidimArray<RDOUBLE> &Iexp, float3 *angles, idxtype batch, MultidimArray<RDOUBLE>* Itheo, MultidimArray<RDOUBLE>* Icorr, MultidimArray<RDOUBLE>* Idiff, MultidimArray<RDOUBLE>* Inorm, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE ART_single_image(const MultidimArray<RDOUBLE> &Iexp, RDOUBLE rot, RDOUBLE tilt, RDOUBLE psi, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE ART_single_image(const MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &Itheo, MultidimArray<RDOUBLE> &Icorr, MultidimArray<RDOUBLE> &Idiff, RDOUBLE rot, RDOUBLE tilt, RDOUBLE psi, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE ART_batched(const MultidimArray<RDOUBLE> &Iexp, idxtype batchSize, float3 *angles, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE ART_batched(const MultidimArray<RDOUBLE> &Iexp, MultidimArray<RDOUBLE> &Itheo, MultidimArray<RDOUBLE> &Icorr, MultidimArray<RDOUBLE> &Idiff, idxtype batchSize, float3 *angles, RDOUBLE shiftX, RDOUBLE shiftY);

	RDOUBLE ART_multi_Image_step(std::vector< MultidimArray<RDOUBLE> > Iexp, std::vector<float3> angles, RDOUBLE shiftX, RDOUBLE shiftY);
	RDOUBLE ART_multi_Image_step(RDOUBLE * Iexp, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE border, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
	RDOUBLE ART_multi_Image_step(RDOUBLE * Iexp, float3 * angles, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
	RDOUBLE ART_multi_Image_step_DB(RDOUBLE * Iexp, RDOUBLE * Itheo, RDOUBLE * Icorr, RDOUBLE * Idiff, float3 * angles, RDOUBLE *gaussTables, RDOUBLE *gaussTables2, RDOUBLE tableLength, RDOUBLE shiftX, RDOUBLE shiftY, unsigned int numImages);
	void projectForward(float3 *angles, idxtype numAngles, MultidimArray<RDOUBLE>& Itheo, RDOUBLE shiftX, RDOUBLE shiftY);

	MRCImage<RDOUBLE> *create3DImage(RDOUBLE oversampling = 1.0);

	void project_Pseudo_batch(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj,
		std::vector<Matrix2D<RDOUBLE>> &EulerVec, RDOUBLE shiftX, RDOUBLE shiftY, int direction);

	void project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj,
		Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY,
		int direction);

	void project_Pseudo(MultidimArray<RDOUBLE> &proj, MultidimArray<RDOUBLE> &norm_proj, std::vector<Matrix1D<RDOUBLE>> * atomPositions,
		Matrix2D<RDOUBLE> &Euler, RDOUBLE shiftX, RDOUBLE shiftY,
		int direction);

	void project_PseudoCTF(RDOUBLE * out, RDOUBLE * out_nrm, RDOUBLE *gaussTable, RDOUBLE * gaussTable2, RDOUBLE border,
		float3 Euler, RDOUBLE shiftX, RDOUBLE shiftY,
		int direction);
	void project_Pseudo(RDOUBLE * out,
						RDOUBLE * out_nrm,
		                float3 angles,
						RDOUBLE shiftX,
						RDOUBLE shiftY,
		                int direction);

	void setAtomPositions(std::vector<float3> newPositions) {
		atoms.AtomPositions = newPositions;
	}

	void writePDB(FileName outpath);

};

#endif // ! PSEUDO_PROJECTOR

