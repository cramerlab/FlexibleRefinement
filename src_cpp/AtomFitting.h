#pragma once

#include "Prerequisites.cuh"

typedef struct {
	double alpha;
	double beta1;
	double beta2;
	double epsilon;
	int t;
	float3 * m;
	float3 * v;

}ADAMParams;