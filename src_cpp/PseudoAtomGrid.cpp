#include "PseudoAtomGrid.h"

void PseudoAtomGrid::initializeGrid()
{
	grid = (std::vector<int>***)malloc(dims.z * sizeof(*grid));
	for (size_t zz = 0; zz < dims.z; zz++)
	{
		grid[zz] = (std::vector<int>**)malloc(dims.y * sizeof(*(grid[zz])));
		for (size_t yy = 0; yy < dims.y; yy++)
		{
			grid[zz][yy] = (std::vector<int>*)malloc(dims.y * sizeof(std::vector<int>));
			for (size_t xx = 0; xx < dims.x; xx++)
				grid[zz][yy][xx] = std::vector<int>();
		}
	}
	for (size_t i = 0; i < AtomPositions.size(); i++)
	{
		int X0 = (int)AtomPositions[i].x;
		int Y0 = (int)AtomPositions[i].y;
		int Z0 = (int)AtomPositions[i].z;
		grid[Z0][Y0][X0].emplace_back(i);
	}
}
