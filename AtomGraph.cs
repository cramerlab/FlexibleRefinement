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

        public float3 Pos { get => pos; set => pos = value; }
        public List<Atom> Neighbours { get => neighbours; set => neighbours = value; }
    }

    struct GridCell
    {
        List<Atom> atoms;
        public List<Atom> Atoms
        {
            get
            {
                if (atoms == null)
                    atoms = new List<Atom>();
                return atoms;
            }
            set => atoms = value; }
        // All Atoms in this grid cell
    }
    float3 gridSpacing;
    int3 gridSize;
    GridCell[][] grid;
    List<Atom> atoms;
    public void SetupNeighbors()
    {
        float cutoff = 5;
        float GridDiag = (float)(Math.Pow(gridSpacing.X, 2) + Math.Pow(gridSpacing.Z, 2) + Math.Pow(gridSpacing.Y, 2));
        float cutoffS = (float)(Math.Pow(cutoff, 2));
        int deltaX = (int)Math.Ceiling(cutoff / gridSpacing.X);
        int deltaY = (int)Math.Ceiling(cutoff / gridSpacing.Y);
        int deltaZ = (int)Math.Ceiling(cutoff / gridSpacing.Z);
        foreach (var atom in atoms)
        {
            int3 gridPos = new int3((int)(atom.Pos.X / gridSpacing.X), (int)(atom.Pos.Y / gridSpacing.Y), (int)(atom.Pos.Z / gridSpacing.Z));
            GridCell start = grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X];
            for (int x = 0; x < deltaX; x++)
            {
                if(gridPos.X + x >= gridSize.X)
                    continue;
                for (int y = 0; y < deltaY; y++)
                {
                    if (gridPos.Y + y >= gridSize.Y)
                        continue;
                    for (int z = 0; z < deltaZ; z++)
                    {
                        //float gridDist = (float)(Math.Pow(start..X, 2) + Math.Pow(gridSpacing.Z, 2) + Math.Pow(gridSpacing.Y, 2));
                        if (gridPos.Z + z >= gridSize.Z)
                            continue;
                        GridCell end = grid[gridPos.Z + z][(gridPos.Y + y) * gridSize.X + gridPos.X + x];
                        foreach (var atomFar in end.Atoms)
                        {
                            float atomDist = (float)((Math.Pow(atomFar.Pos.X - atom.Pos.X, 2) + Math.Pow(atomFar.Pos.Y - atom.Pos.Y, 2) + Math.Pow(atomFar.Pos.Z - atom.Pos.Z, 2)));
                            if (atomDist <= cutoffS && atomFar.Pos != atom.Pos)
                            {
                                atom.Neighbours.Add(atomFar);
                                atomFar.Neighbours.Add(atom);
                            }
                        }
                    }
                }
            }
        }
    }

    public AtomGraph(float3[] positions, float[] r, int3 dim)
	{
        if (positions.Length != r.Length)
            throw new Exception("Error");
        gridSpacing = new float3(1.0f);
        gridSize = new int3((int)(dim.X / gridSpacing.X), (int)(dim.Y / gridSpacing.Y), (int)(dim.Z / gridSpacing.Z)); //actual (rounded) grid size in cells
        gridSpacing = new float3(dim.X / gridSize.X, dim.Y / gridSize.Y, dim.Z / gridSize.Z); // actual spacing now that number of cells is known

        atoms = new List<Atom>();
        grid = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j=> new GridCell(),gridSize.X* gridSize.Y), dim.Z);


        for (int i = 0; i < positions.Length; i++)
        {
            float3 pos = positions[i];
            int3 gridPos = new int3((int)(pos.X / gridSpacing.X), (int)(pos.Y / gridSpacing.Y), (int)(pos.Z / gridSpacing.Z));
            Atom a = new Atom(pos, r[i]);
            atoms.Add(a);
            grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X].Atoms.Add(a);

        }
	}
}
