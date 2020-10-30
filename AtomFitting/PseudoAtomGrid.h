#pragma once
#include "pseudoatoms.h"
class PseudoAtomGrid :
	public Pseudoatoms
{
	std::vector<int> *** grid;
	int3 dims;
	template<typename ... Args>
	PseudoAtomGrid(int3 dims, Args... args) : Pseudoatoms(args...), dims(dims) {
		initializeGrid();
	}

	void initializeGrid();

};

