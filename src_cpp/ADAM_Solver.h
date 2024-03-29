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

	void run(AtomMover &mover, int numIt, FileName outfile="");
	ADAM_Solver(double Alpha = 0.01, double Beta1 = 0.9, double Beta2 = 0.999, double Epsilon = 1e-8) : alpha(Alpha), beta1(Beta1), beta2(Beta2), epsilon(Epsilon), t(0) {};
};

