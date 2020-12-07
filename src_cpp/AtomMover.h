#pragma once
#include "Eigen\Core"
#include "liblionImports.h"
#include "Types.h"
#include "funcs.h"
#include "Warp_GPU.h"
#include "pseudoatoms.h"



class AtomMover {
public:
	Pseudoatoms *Atoms;
	MultidimArray<RDOUBLE> SuperRefVolume;
	int3 Dims;
	RDOUBLE Super;
	bool Resize;
	bool Weighting;
	float K; //constant for elastic movement of atoms from their initial configuration

	AtomMover(Pseudoatoms *atoms, MultidimArray<RDOUBLE>& superRefVol, int3 dims, RDOUBLE super, bool resize, bool weighting, float k) :Atoms(atoms), SuperRefVolume(superRefVol), Dims(dims), Super(super), Resize(resize), Weighting(weighting), K(k) {}
	void run();

	double operator()(Eigen::VectorXd positions, Eigen::VectorXd& grad);
};
