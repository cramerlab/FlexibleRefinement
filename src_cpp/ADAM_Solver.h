#pragma once
#include "AtomMover.h"
#include "Eigen\Core"
class ADAM_Solver
{

		double alpha;
		double beta1;
		double beta2;
		double epsilon;
		int t;
		Eigen::VectorXd m;
		Eigen::VectorXd v;

public:
	void run(AtomMover &mover, int numIt);
};

