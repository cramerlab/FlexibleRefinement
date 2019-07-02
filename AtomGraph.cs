using System;
using Warp.Tools;
using Warp;
using System.Collections.Generic;

public class AtomGraph
{
    struct Atom
    {
        public Atom(float3 p, float radius)
        {
            pos = p;
            r = radius;
            neighbours = new List<Atom>();
        }
        float3 pos;             // atom position in real space
        List<Atom> neighbours;      // 
        float r;
    }

    struct GridCell
    {
        List<Atom> atoms;
        public List<Atom> Atoms { get => atoms; set => atoms = value; }
        // All Atoms in this grid cell
    }

    public AtomGraph(float3[] positions, float[] r, int3 dim)
	{
        if (positions.Length != r.Length)
            throw new Exception("Error");
        float3 gridSpacing = new float3(1.0f);
        int3 gridSize = new int3((int)(dim.X / gridSpacing.X), (int)(dim.Y / gridSpacing.Y), (int)(dim.Z / gridSpacing.Z)); //actual (rounded) grid size in cells
        gridSpacing = new float3(dim.X / gridSize.X, dim.Y / gridSize.Y, dim.Z / gridSize.Z); // actual spacing now that number of cells is known
        
        
        GridCell[][] grid = Helper.ArrayOfFunction(i => new GridCell[gridSize.X* gridSize.Y], dim.Z);


        for (int i = 0; i < positions.Length; i++)
        {
            float3 pos = positions[i];
            int3 gridPos = new int3((int)(pos.X / gridSpacing.X), (int)(pos.Y / gridSpacing.Y), (int)(pos.Z / gridSpacing.Z));
            grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X].Atoms.Add(new Atom(pos, r[i]));

        }
	}
}
