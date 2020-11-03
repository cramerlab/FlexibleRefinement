#pragma once
#include "Eigen\Core"
#include "liblionImports.h"
#include "Types.h"
#include "funcs.h"
#include "Warp_GPU.h"
#include "pseudoatoms.h"



class Solver {

	Pseudoatoms *Atoms;
	MultidimArray<RDOUBLE> SuperRefVolume;
	int3 Dims;
	RDOUBLE Super;
	bool Resize;

public:
	Solver(Pseudoatoms *atoms, MultidimArray<RDOUBLE>& superRefVol, int3 dims, RDOUBLE super, bool resize) :Atoms(atoms), SuperRefVolume(superRefVol), Dims(dims), Super(super), Resize(resize) {}
	void run();

	double operator()(Eigen::VectorXd positions, Eigen::VectorXd& grad);
};
