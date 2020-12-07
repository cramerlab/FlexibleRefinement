#pragma once
#include "AtomMover.h"
class LBFGS_Solver
{
public:
	void run(AtomMover &mover, int numIt);
};

