using System;
using Warp.Tools;
using Warp;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.IO;
using System.Text;
using System.Linq;
using System.Globalization;
using System.Diagnostics;

namespace FlexibleRefinement.Util
{

    public class Atom
    {
        public Atom(float3 p, float radius, float intense = 1.0f)
        {
            pos = p;
            r = radius;
            neighbours = new List<Atom>();
            intensity = intense;
        }
        float3 pos;             // atom position in real space
        List<Atom> neighbours;      // 
        float r;
        float intensity;

        public float3 Pos { get => pos; set => pos = value; }
        public float Intensity { get => intensity; set => intensity = value; }
        public List<Atom> Neighbours { get => neighbours; set => neighbours = value; }
        public float R { get => r; set => r = value; }
    }

    public class GridCell
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
            set => atoms = value;
        }
        // All Atoms in this grid cell
    }

    public class AtomGraph
    {

        public List<Atom> Atoms { get => atoms; }
        public float NeigbourCutoff { get => neigbourCutoff; set => neigbourCutoff = value; }
        public double R0 { get => r0; set => r0 = value; }

        float3 gridSpacing;
        int3 gridSize;
        GridCell[][] grid;
        List<Atom> atoms;
        Image EMIntensities;
        IntPtr einSpline;
        float neigbourCutoff = 4.0f;
        double r0 = 3.0;
        float R0_6;



        public float3[][] CalculateForces(float3[][] grad)
        {
            float3[][] forces = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new float3(0), gridSize.X * gridSize.Y), gridSize.Z);

            float3 alpha = new float3(8.0f);
            for (int x = 0; x < gridSize.X; x++)
            {

                for (int y = 0; y < gridSize.Y; y++)
                {
                    for (int z = 0; z < gridSize.Z; z++)
                    {
                        forces[z][y * gridSize.X + x] = alpha * grad[z][y * gridSize.X + x];
                    }

                }
            }
            return forces;
        }

        public float3 CalculateForce(Atom a)
        {
            float3 totForce = new float3(0);
            float3 intForce = IntensityF(a.Pos);
            totForce = totForce + intForce;
            return totForce;
        }

        public void setEMIntensities(Image intensities)
        {
            EMIntensities = intensities.GetCopy();
            einSpline = ImageToSpline(EMIntensities);
        }


        public Image Repr(int threads = 1)
        {
            Image rep = new Image(Dim);
            float[][] repData = rep.GetHost(Intent.Read);
            foreach (var atom in atoms)
            {
                for (int z = (int)Math.Floor(atom.Pos.Z-atom.R); z <= (int)Math.Ceiling(atom.Pos.Z + atom.R); z++)
                {
                    for (int y = (int)Math.Floor(atom.Pos.Y - atom.R); y <= (int)Math.Ceiling(atom.Pos.Y + atom.R); y++)
                    {
                        for (int x = (int)Math.Floor(atom.Pos.X - atom.R); x <= (int)Math.Ceiling(atom.Pos.X + atom.R); x++)
                        { 
                            double dist = Math.Pow(atom.Pos.X - x, 2) + Math.Pow(atom.Pos.Y - y, 2) + Math.Pow(atom.Pos.Z - z, 2);
                            repData[z][Dim.X * y + x] += dist < atom.R ? 1 : 0;
                        }

                    }
                }
            }
            return rep;
        }

        public Image Repr(double sigma, int threads = 1)
        {
            Image rep = new Image(Dim);
            double sigSqrd = Math.Pow(sigma, 2);
            float[][] repData = rep.GetHost(Intent.Read);
            foreach (var atom in atoms)
            {
                for (int z = (int)Math.Floor(atom.Pos.Z - 3*sigma); z <= (int)Math.Ceiling(atom.Pos.Z + 3 * sigma); z++)
                {
                    for (int y = (int)Math.Floor(atom.Pos.Y - 3 * sigma); y <= (int)Math.Ceiling(atom.Pos.Y + 3 * sigma); y++)
                    {
                        for (int x = (int)Math.Floor(atom.Pos.X - 3 * sigma); x <= (int)Math.Ceiling(atom.Pos.X + 3 * sigma); x++)
                        {
                            double r = Math.Pow(z - atom.Pos.Z, 2) + Math.Pow(y - atom.Pos.Y, 2) + Math.Pow(x - atom.Pos.X, 2);
                            repData[z][Dim.X * y + x] += (float)Math.Exp(-r/sigSqrd);
                        }

                    }
                }
            }
            return rep;
        }

        public Image IntRepr()
        {
            Image rep = new Image(Dim);
            float[][] repData = rep.GetHost(Intent.Read);
            Helper.ForCPU(0, Dim.Z, 20, null, (z, id, ts) =>
            {
                for (int y = 0; y < Dim.Y; y++)
                {
                    for (int x = 0; x < Dim.X; x++)
                    {
                        repData[z][Dim.X * y + x] = getIntensity(new float3(x, y, z));
                    }
                }
            }, null);
            return rep;
        }


        private float3 DistF(float3 r1, float3 r2)
        {

            float3 vecR = (r2 - r1);
            double dist = Math.Max(vecR.Length(), 1e-6f);
            //return vecR * (float)(-2 * (dist - R0) / dist);
            //return vecR * (float)(12 * R0_6 / Math.Pow(dist, 8) * (1 - Math.Pow(R0/dist, 6)));
            return vecR * (float)Math.Round(((12 * Math.Pow(R0, 6) / Math.Pow(dist, 8) * (1 - Math.Pow(R0 / dist, 6)))),3);
        }

        private float3 DistF(Atom atom)
        {
            float3 force = new float3(0);

            foreach (var btom in atom.Neighbours)
            {
                force = force + DistF(atom.Pos, btom.Pos);
            }

            return force;
        }

        private float3 DistFSqrd(float3 r1, float3 r2)
        {

            float3 vecR = (r2 - r1);
            float len = vecR.Length();
            double dist = len - R0;
            //return vecR * (float)(-2 * (dist - R0) / dist);
            return new float3((float)(vecR.X / len * 2 * dist), (float)(vecR.Y / len * 2 * dist), (float)(vecR.Z / len * 2 * dist));
        }

        private float3 DistFSqrd(Atom atom)
        {
            float3 force = new float3(0);

            foreach (var btom in atom.Neighbours)
            {
                force = force + DistFSqrd(atom.Pos, btom.Pos);
            }

            return force;
        }


        private float3 IntensityF(float3 pos)
        {
            float stepsize = 0.1f;
            float invStepsize = 1.0f / stepsize;
            float3 f = new float3(0);
            float int0 = getIntensity(pos);
            for (int x = -1; x <= 1; x++)
            {
                if (pos.X + x * stepsize >= gridSize.X || pos.X + x * stepsize <= 0)
                    continue;
                for (int y = -1; y <= 1; y++)
                {
                    if (pos.Y + y * stepsize >= gridSize.Y || pos.Y + y * stepsize <= 0)
                        continue;
                    for (int z = -1; z <= 1; z++)
                    {

                        if (pos.Z + z * stepsize >= gridSize.Z || pos.Z + z * stepsize <= 0)
                            continue;
                        float int1 = getIntensity(pos + new float3(x * stepsize, y * stepsize, z * stepsize));
                        f = f + new float3(x * invStepsize * (int1 - int0), y * invStepsize * (int1 - int0), z * invStepsize * (int1 - int0));

                    }
                }
            }
            return f;
        }

        private float3 CorrForce(Atom atom, float sumAs, float sumIs, float sumAI)
        {
            float correlation0 = (float)(sumAI / (sumAs * sumIs));
            // Calculate this atoms contribution

            float As0 = atom.Intensity * atom.Intensity;
            float AI0 = atom.Intensity * getIntensity(atom.Pos);
            float3 f = new float3(0);
            float stepsize = 0.1f;
            float invStepsize = (float)(1.0f / stepsize);

            for (int x = -1; x <= 1; x++)
            {
                if (atom.Pos.X + x * stepsize >= gridSize.X || atom.Pos.X + x * stepsize <= 0)
                    continue;
                for (int y = -1; y <= 1; y++)
                {
                    if (atom.Pos.Y + y * stepsize >= gridSize.Y || atom.Pos.Y + y * stepsize <= 0)
                        continue;
                    for (int z = -1; z <= 1; z++)
                    {

                        if (atom.Pos.Z + z * stepsize >= gridSize.Z || atom.Pos.Z + z * stepsize <= 0)
                            continue;
                        float AI1 = atom.Intensity * getIntensity(atom.Pos + new float3(x * stepsize, y * stepsize, z * stepsize));
                        float As1 = As0;
                        float correlation1 = (float)((sumAI - AI0 + AI1) / ((sumAs - As0 + As1) * sumIs));
                        float diff = AI1 - AI0;
                        f = f + new float3(x * invStepsize * (diff), y * invStepsize * (diff), z * invStepsize * (diff));
                    }
                }
            }

            return f;
        }

        private void CurrentCorrelation(out float sumAs, out float sumIs, out float sumAI)
        {
            sumAs = 0;
            sumIs = 0;
            sumAI = 0;

            foreach (var a in atoms)
            {
                sumAs += (float)Math.Pow(a.Intensity, 2);
                sumAI += a.Intensity * getIntensity(a.Pos);
            }
            Image tmp = EMIntensities.GetCopy();
            tmp.Multiply(tmp);
            sumIs = tmp.AsSum3D().GetHost(Intent.Read)[0][0];
            tmp.Dispose();
        }


        private void MoveAtom(Atom atom, float3 ds)
        {
            int3 gridPosBefore = new int3((int)(atom.Pos.X / gridSpacing.X), (int)(atom.Pos.Y / gridSpacing.Y), (int)(atom.Pos.Z / gridSpacing.Z));
            GridCell gridCellBefore = grid[gridPosBefore.Z][gridPosBefore.Y * gridSize.X + gridPosBefore.X];

            double delta = Math.Sqrt(Math.Pow(ds.X, 2) + Math.Pow(ds.Y, 2) + Math.Pow(ds.Z, 2));

           
            double movementCutoff = 0.1;
            if (delta > movementCutoff)
            {
                ds = new float3((float)(movementCutoff * (ds.X / delta)), (float)(movementCutoff * (ds.Y / delta)), (float)(movementCutoff * (ds.Z / delta)));
            }
            atom.Pos = atom.Pos + ds;
            if (atom.Pos.X < 0)
                atom.Pos = atom.Pos * new float3(0, 1, 1);
            else if (atom.Pos.X > Dim.X)
                atom.Pos = new float3(Dim.X - 1e-4f, atom.Pos.Y, atom.Pos.Z);
            if (atom.Pos.Y < 0)
                atom.Pos = atom.Pos * new float3(1, 0, 1);
            else if (atom.Pos.Y > Dim.Y)
                atom.Pos = new float3(atom.Pos.X, Dim.Y - 1e-4f, atom.Pos.Z);
            if (atom.Pos.Z < 0)
                atom.Pos = atom.Pos * new float3(1, 1, 0);
            else if (atom.Pos.Z > Dim.Z)
                atom.Pos = new float3(atom.Pos.X, atom.Pos.Y, Dim.Z - 1e-4f);
            int3 gridPosAfter = new int3((int)(atom.Pos.X / gridSpacing.X), (int)(atom.Pos.Y / gridSpacing.Y), (int)(atom.Pos.Z / gridSpacing.Z));
            if (gridPosAfter != gridPosBefore)
            {
                GridCell gridCellAfter = grid[gridPosAfter.Z][gridPosAfter.Y * gridSize.X + gridPosAfter.X];
                gridCellBefore.Atoms.Remove(atom);
                gridCellAfter.Atoms.Add(atom);
                //TODO: update neigbors as well
            }
        }

        public void moveAtoms(float3[][] forcefield)
        {

            float[] xForce = new float[Dim.Elements()];
            float[] yForce = new float[Dim.Elements()];
            float[] zForce = new float[Dim.Elements()];
            for (int z = 0; z < Dim.Z; z++)
            {
                for (int x = 0; x < Dim.X; x++)
                {
                    for (int y = 0; y < Dim.Y; y++)
                    {
                        xForce[(x * Dim.Y + y) * Dim.Z + z] = forcefield[z][y * Dim.X + x].X;
                        yForce[(x * Dim.Y + y) * Dim.Z + z] = forcefield[z][y * Dim.X + x].Y;
                        zForce[(x * Dim.Y + y) * Dim.Z + z] = forcefield[z][y * Dim.X + x].Z;
                        if(float.IsNaN(forcefield[z][y * Dim.X + x].X) || float.IsNaN(forcefield[z][y * Dim.X + x].Y) || float.IsNaN(forcefield[z][y * Dim.X + x].Z))
                        {
                            Console.WriteLine("test");
                        }

                    }
                }
            }
            IntPtr xForceSpline = CPU.CreateEinspline3(xForce, Dim, new float3(0));
            IntPtr yForceSpline = CPU.CreateEinspline3(yForce, Dim, new float3(0));
            IntPtr zForceSpline = CPU.CreateEinspline3(zForce, Dim, new float3(0));

            float[] tmp = new float[1];

            foreach (var atom in Atoms)
            {

                float3 force = new float3(0);
                CPU.EvalEinspline3(xForceSpline, new float[] { (float)(atom.Pos.Z / Dim.Z), (float)(atom.Pos.Y / Dim.Y), (float)(atom.Pos.X / Dim.X) }, 1, tmp);
                force.X = tmp[0];

                CPU.EvalEinspline3(yForceSpline, new float[] { (float)(atom.Pos.Z / Dim.Z), (float)(atom.Pos.Y / Dim.Y), (float)(atom.Pos.X / Dim.X) }, 1, tmp);
                force.Y = tmp[0];
                CPU.EvalEinspline3(zForceSpline, new float[] { (float)(atom.Pos.Z / Dim.Z), (float)(atom.Pos.Y / Dim.Y), (float)(atom.Pos.X / Dim.X) }, 1, tmp);
                force.Z = tmp[0];

                float3 distForce = DistF(atom);
                MoveAtom(atom, force + distForce);

            }

        }

        public void moveAtoms(float corrScale = 20.0f, float distScale = 1.0f, bool normalizeForce = true)
        {
            CurrentCorrelation(out float sumAs, out float sumIs, out float sumAI);
            foreach (var atom in atoms)
            {
                float3 intensityForce = IntensityF(atom.Pos);
                float intensityLength = intensityForce.Length();
                float3 distForce = new float3(0);
                if (atom.Neighbours.Count == 0)
                {   //Make sure that a single atom will not loose the connection to the rest
                    Atom b = GetClosestAtom(atom);
                    atom.Neighbours.Add(b);
                }

                distForce = DistF(atom) * distScale;

                float3 corrForce = CorrForce(atom, sumAs, sumIs, sumAI) * corrScale;

                if (normalizeForce)
                {
                    if (corrForce.Length() > 1.0f)
                    {
                        corrForce = corrForce / corrForce.Length();
                    }
                    double distL = Math.Sqrt(Math.Pow(distForce.X, 2) + Math.Pow(distForce.Y, 2) + Math.Pow(distForce.Z, 2));
                    if (distL > 1.0f)
                    {
                        distForce = new float3((float)(((double)distForce.X) / distL), (float)(((double)distForce.Y) / distL), (float)(((double)distForce.Z) / distL));
                    }
                }
                //Contribution to correlation before moving
                float As0 = atom.Intensity * atom.Intensity;
                float AI0 = atom.Intensity * getIntensity(atom.Pos);

                if ((corrForce + distForce).Length() == 0)
                    Console.WriteLine("Offset is 0");

                MoveAtom(atom, corrForce + distForce);

                //contribution to correlation after moving
                float As1 = atom.Intensity * atom.Intensity;
                float AI1 = atom.Intensity * getIntensity(atom.Pos);

                sumAs = sumAs - As0 + As1;
                sumAI = sumAI - AI0 + AI1;
                /* TODO: Not updating neighbours for now
                List<Atom> newNeighbours = getNeighbours(atom, false);



                foreach (var btom in atom.Neighbours)
                {
                    if (!newNeighbours.Contains(btom))
                    {
                        btom.Neighbours.Remove(atom);
                    }
                }
                atom.Neighbours = newNeighbours;
                foreach (var btom in atom.Neighbours)
                {
                    if (!btom.Neighbours.Contains(atom))
                    {
                        btom.Neighbours.Add(atom);
                    }
                }
                */
            }
        }


        public void moveAtoms(float3[][] forces, float scale = 1.0f)
        {

            Helper.ForCPU(0, atoms.Count, 20, null, (i, id, obj) =>
               {
                   Atom atom = atoms[i];
                   float stepSize = 0.1f;
                   float3 apos = atom.Pos;
                   float3 distForce = new float3(0.0f);
                   float beta = 1.0f / 1.0f;
                   foreach (var btom in atom.Neighbours)
                   {
                           /*
                           float3 bpos = btom.Pos;
                           float E0 = DistE((bpos - apos).Length());
                           for (int x = -1; x <= 1; x++)
                           {
                               for (int y = -1; y <= 1; y++)
                               {
                                   for (int z = -1; z <= 1; z++)
                                   {
                                       float3 shifted = new float3(apos.X + x * stepSize, apos.Y + y * stepSize, apos.Z + z * stepSize);
                                       float E1 = DistE((bpos - shifted).Length());
                                       float diffQuot = (E1 - E0) / stepSize;
                                       distForce += new float3(x * diffQuot, y * diffQuot, z * diffQuot);
                                   }
                               }
                           }
                           */
                       distForce = distForce + DistF(apos, btom.Pos);
                   }
                   distForce = distForce * (beta);
                   float3 imForce = forces[(int)Math.Round(atom.Pos.Z)][(int)((int)(Math.Round(atom.Pos.Y)) * gridSize.X + (int)(Math.Round(atom.Pos.X)))];
                   MoveAtom(atom, distForce + imForce);
               }, null);
        }

        public void SetupNeighbors()
        {
            foreach (var atom in atoms)
            {
                List<Atom> neighbours = getNeighbours(atom, true);
                foreach (var btom in neighbours)
                {
                    btom.Neighbours.Add(atom);
                    atom.Neighbours.Add(btom);
                }

            }

            int[] numNeig = Helper.ArrayOfFunction(i => atoms[i].Neighbours.Count, atoms.Count);
        }

        //Returns all Atoms whose distance to center is smaller than cutoff
        public List<Atom> getNeighbours(Atom atom, bool halfOnly = false)
        {
            float3 center = atom.Pos;
            float GridDiag = (float)(Math.Pow(gridSpacing.X, 2) + Math.Pow(gridSpacing.Z, 2) + Math.Pow(gridSpacing.Y, 2));
            float cutoffS = (float)(Math.Pow(neigbourCutoff, 2));
            int deltaX = (int)Math.Ceiling(neigbourCutoff / gridSpacing.X);
            int deltaY = (int)Math.Ceiling(neigbourCutoff / gridSpacing.Y);
            int deltaZ = (int)Math.Ceiling(neigbourCutoff / gridSpacing.Z);
            List<Atom> neighbours = new List<Atom>();

            int3 gridPos = new int3((int)(center.X / gridSpacing.X), (int)(center.Y / gridSpacing.Y), (int)(center.Z / gridSpacing.Z));
            GridCell start = grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X];
            for (int x = -deltaX; x < deltaX; x++)
            {
                if (gridPos.X + x >= gridSize.X || gridPos.X + x < 0)
                    continue;
                for (int y = -deltaY; y < deltaY; y++)
                {
                    if (gridPos.Y + y >= gridSize.Y || gridPos.Y + y < 0)
                        continue;
                    for (int z = halfOnly ? 0 : -deltaZ; z < deltaZ; z++)
                    {
                        //float gridDist = (float)(Math.Pow(start..X, 2) + Math.Pow(gridSpacing.Z, 2) + Math.Pow(gridSpacing.Y, 2));
                        if (gridPos.Z + z >= gridSize.Z || gridPos.Z + z < 0)
                            continue;
                        GridCell end = grid[gridPos.Z + z][(gridPos.Y + y) * gridSize.X + gridPos.X + x];
                        foreach (var atomFar in end.Atoms)
                        {
                            float atomDist = (float)((Math.Pow(atomFar.Pos.X - center.X, 2) + Math.Pow(atomFar.Pos.Y - center.Y, 2) + Math.Pow(atomFar.Pos.Z - center.Z, 2)));
                            if (atomDist <= cutoffS && atomFar != atom)
                            {
                                neighbours.Add(atomFar);
                            }
                        }
                    }
                }
            }
            return neighbours;
        }



        private int3 Dim;

        //initialize placement of atoms in grid cells
        private void InitializeAtomGrid(float3[] atomCenters, float[] atomRadius, float[] atomIntensities = null)
        {
            gridSpacing = new float3(1.0f);
            gridSize = new int3((int)(Dim.X / gridSpacing.X), (int)(Dim.Y / gridSpacing.Y), (int)(Dim.Z / gridSpacing.Z)); //actual (rounded) grid size in cells
            gridSpacing = new float3(Dim.X / gridSize.X, Dim.Y / gridSize.Y, Dim.Z / gridSize.Z); // actual spacing now that number of cells is known

            atoms = new List<Atom>();
            grid = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new GridCell(), gridSize.X * gridSize.Y), Dim.Z);


            for (int i = 0; i < atomCenters.Length; i++)
            {
                float3 pos = atomCenters[i];
                int3 gridPos = new int3((int)(pos.X / gridSpacing.X), (int)(pos.Y / gridSpacing.Y), (int)(pos.Z / gridSpacing.Z));
                Atom a = new Atom(pos, atomRadius[i], (atomIntensities != null) ? atomIntensities[i] : 1.0f);
                atoms.Add(a);
                grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X].Atoms.Add(a);

            }
            SetupNeighbors();
        }

        private float getIntensity(float3 pos)
        {
            float[] output = new float[1];
            CPU.EvalEinspline3(einSpline, new float[] { (float)(pos.Z / Dim.Z), (float)(pos.Y / Dim.Y), (float)(pos.X / Dim.X) }, 1, output);
            return output[0];
        }

        private float[] getIntensity(float3[] pos)
        {
            float[] output = new float[pos.Length];
            float[] input = new float[pos.Length * 3];
            for (int i = 0; i < pos.Length; i++)
            {
                input[i] = pos[i].X;
                input[i + 1] = pos[i].Y;
                input[i + 2] = pos[i].Z;
            }
            CPU.EvalEinspline3(einSpline, input, pos.Length, output);
            return output;
        }

        private IntPtr ImageToSpline(Image im)
        {
            float[][] data = im.GetHost(Intent.Read);
            float[] values = new float[im.ElementsReal];

            for (int z = 0; z < im.Dims.Z; z++)
            {
                for (int y = 0; y < im.Dims.Y; y++)
                {
                    for (int x = 0; x < im.Dims.X; x++)
                    {
                        values[(x * im.Dims.Y + y) * im.Dims.Z + z] = data[z][y * im.Dims.X + x];
                    }
                }

            }

            return CPU.CreateEinspline3(values, im.Dims, new float3(0));
        }

        private static void AddText(FileStream fs, string value)
        {
            byte[] info = new UTF8Encoding(true).GetBytes(value);
            fs.Write(info, 0, info.Length);
        }

        public Atom GetClosestAtom(Atom a)
        {
            return GetClosestAtom(a.Pos);
        }

        public Atom GetClosestAtom(float3 pos)
        {
            int3 gridPos = new int3((int)(pos.X / gridSpacing.X), (int)(pos.Y / gridSpacing.Y), (int)(pos.Z / gridSpacing.Z));
            GridCell start = grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X];
            bool checkedOne;
            int i = 0;
            int iLim = int.MaxValue;
            float minDist = float.PositiveInfinity;
            Atom minAtom = null;
            do
            {
                checkedOne = false;
                for (int dz = -i; dz <= i; dz++)
                {
                    if (gridPos.Z + dz >= gridSize.Z)
                        continue;
                    for (int dy = -i; dy <= i; dy++)
                    {
                        if (gridPos.Y + dy >= gridSize.Y)
                            continue;
                        for (int dx = -i; dx <= i; dx++)
                        {
                            if (gridPos.X + dx >= gridSize.X)
                                continue;
                            if (!(Math.Abs(dx) == i || Math.Abs(dy) == i || Math.Abs(dz) == i))
                                continue;
                            GridCell curr = grid[gridPos.Z + dz][(gridPos.Y + dy) * gridSize.X + gridPos.X + dx];
                            checkedOne = true; //GridCells could be found
                            if (curr.Atoms.Count > 0)
                            {
                                iLim = Math.Min(iLim, i + 1); // In current cubic layer there are atoms, therefore we can terminate with next layer
                                foreach (var atom in curr.Atoms)
                                {
                                    float dist = (atom.Pos - pos).Length();
                                    if (dist < minDist)
                                    {
                                        minAtom = atom;
                                        minDist = dist;
                                    }
                                }
                            }
                        }
                    }
                }
                i++;
            } while (checkedOne && i <= iLim);
            return minAtom;
        }

        /*
         * Updates the atom Positions in <this> based on a nearest neighbour search in the graph <other> and displacements <displ> thereof
         * */
        public void setPositions(AtomGraph other, List<float3> displ)
        {
            float3 scaleFactor = new float3((float)Dim.X / other.Dim.X, (float)Dim.Y / other.Dim.Y, (float)Dim.Z / other.Dim.Z);
            foreach (var atom in Atoms)
            {
                Atom nN = other.GetClosestAtom(atom.Pos / scaleFactor);
                float3 displacement = displ[other.Atoms.IndexOf(nN)] * scaleFactor;

                atom.Pos = atom.Pos + displacement;
                Debug.Assert(atom.Pos.X <= Dim.X || atom.Pos.Y <= Dim.Y || atom.Pos.Z <= Dim.Z);

            }
        }


        public AtomGraph(String filename, Image intensities)
        {
            int counter = 0;
            string line;
            float[] result;
            
            EMIntensities = intensities;
            if (intensities != null)
            {
                einSpline = ImageToSpline(intensities);
            }
            atoms = new List<Atom>();
            // Read the file and display it line by line.  
            System.IO.StreamReader file = new System.IO.StreamReader(filename);
            Dim = intensities.Dims;
            if (filename.EndsWith(".graph"))
            {
                int[] temp;
                line = file.ReadLine();
                temp = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s)).ToArray();
                Dim = new int3(temp[0], temp[1], temp[2]);

                line = file.ReadLine();
                result = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture.NumberFormat)).ToArray();
                gridSpacing = new float3(result[0], result[1], result[2]);

                line = file.ReadLine();
                temp = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => int.Parse(s)).ToArray();
                gridSize = new int3(temp[0], temp[1], temp[2]);

                line = file.ReadLine();
                neigbourCutoff = float.Parse(line, CultureInfo.InvariantCulture.NumberFormat);

                line = file.ReadLine();
                R0 = float.Parse(line, CultureInfo.InvariantCulture.NumberFormat);

                line = file.ReadLine();
                while ((line = file.ReadLine()) != null)
                {
                    result = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture.NumberFormat)).ToArray();
                    atoms.Add(new Atom(new float3(result[0], result[1], result[2]), result[3], result[4]));
                }
            }
            else if (filename.EndsWith(".xyz"))
            {
                float[] temp;
                line = file.ReadLine();
                //Atom count does not need to be parsed
                line = file.ReadLine();
                temp = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture.NumberFormat)).ToArray();

                gridSpacing = new float3(temp[0], temp[1], temp[2]);
                gridSize = new int3((int)temp[3], (int)temp[4], (int)temp[5]);
                neigbourCutoff = temp[6];
                R0 = temp[7];
                while ((line = file.ReadLine()) != null)
                {
                    line = line.Substring(2);
                    result = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture.NumberFormat)).ToArray();
                    float3 pos = new float3(result[0], result[1], result[2]);
                    atoms.Add(new Atom(pos, (float)(R0/2), getIntensity(pos)));
                }
            }
            file.Close();
            grid = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new GridCell(), gridSize.X * gridSize.Y), Dim.Z);


            foreach (var atom in atoms)
            {


                float3 pos = atom.Pos;
                int3 gridPos = new int3((int)(pos.X / gridSpacing.X), (int)(pos.Y / gridSpacing.Y), (int)(pos.Z / gridSpacing.Z));

                grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X].Atoms.Add(atom);

            }
            SetupNeighbors();


        }

        public void save(String filename)
        {
            FileStream fs = null;
            try
            {
                fs = File.Create(filename);
                if (filename.EndsWith(".graph"))
                {
                
                        AddText(fs, $"{Dim.X} {Dim.Y} {Dim.Z}\n".Replace(',', '.'));
                        AddText(fs, $"{gridSpacing.X} {gridSpacing.Y} {gridSpacing.Z}\n".Replace(',', '.'));
                        AddText(fs, $"{gridSize.X} {gridSize.Y} {gridSize.Z}\n".Replace(',', '.'));
                        AddText(fs, $"{neigbourCutoff}\n".Replace(',', '.'));
                        AddText(fs, $"{R0}\n".Replace(',', '.'));
                        AddText(fs, $"#Atoms\n");
                        foreach (var atom in atoms)
                        {
                            AddText(fs, $"{atom.Pos.X} {atom.Pos.Y} {atom.Pos.Z} {atom.R} {atom.Intensity}\n".Replace(',', '.'));
                        }

                
                }
                else if (filename.EndsWith(".xyz"))
                {
                    AddText(fs, $"{Atoms.Count}\n");
                    AddText(fs, $"{gridSpacing.X} {gridSpacing.Y} {gridSpacing.Z}".Replace(',', '.'));
                    AddText(fs, $" {gridSize.X} {gridSize.Y} {gridSize.Z}".Replace(',', '.'));
                    AddText(fs, $" {neigbourCutoff}".Replace(',', '.'));
                    AddText(fs, $" {R0}\n".Replace(',', '.'));
                    foreach (var atom in atoms)
                    {
                        AddText(fs, $"C {atom.Pos.X} {atom.Pos.Y} {atom.Pos.Z}\n".Replace(',', '.'));
                    }
                }
                else
                    throw new NotImplementedException("This file extensions is not supported for saving");
            }
            catch (IOException)
            {
                Console.WriteLine($"Cannot Save to {filename}, resource is busy");

            }
            if (fs != null)
                fs.Close();
        }


        public AtomGraph(Image intensities, Image mask, int numAtoms = 1000, float r0 = 0.0f)
        {
            float rAtoms = 1.0f;

            EMIntensities = intensities;
            Dim = intensities.Dims;

            GPU.Normalize(intensities.GetDevice(Intent.Read),
                              intensities.GetDevice(Intent.Write),
                              (uint)intensities.ElementsReal,
                              (uint)1);

            einSpline = ImageToSpline(intensities);
            float3[] atomCenters = ImageProcessor.FillWithEquidistantPoints(mask, numAtoms, out rAtoms, r0);
            neigbourCutoff = 2.5f * rAtoms;
            R0 = 2.0f * rAtoms;
            R0_6 = (float)Math.Pow(R0, 6);
            float[] atomRadius = Helper.ArrayOfFunction(i => rAtoms, atomCenters.Length);
            float[] atomIntensities = getIntensity(atomCenters);
            InitializeAtomGrid(atomCenters, atomRadius);
        }

        public AtomGraph(float3[] positions, float[] r, int3 dim)
        {
            if (positions.Length != r.Length)
                throw new Exception("Error");
            Dim = dim;
            gridSpacing = new float3(1.0f);
            gridSize = new int3((int)(dim.X / gridSpacing.X), (int)(dim.Y / gridSpacing.Y), (int)(dim.Z / gridSpacing.Z)); //actual (rounded) grid size in cells
            gridSpacing = new float3(dim.X / gridSize.X, dim.Y / gridSize.Y, dim.Z / gridSize.Z); // actual spacing now that number of cells is known

            atoms = new List<Atom>();
            grid = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new GridCell(), gridSize.X * gridSize.Y), dim.Z);


            for (int i = 0; i < positions.Length; i++)
            {
                float3 pos = positions[i];
                int3 gridPos = new int3((int)(pos.X / gridSpacing.X), (int)(pos.Y / gridSpacing.Y), (int)(pos.Z / gridSpacing.Z));
                Atom a = new Atom(pos, r[i]);
                atoms.Add(a);
                grid[gridPos.Z][gridPos.Y * gridSize.X + gridPos.X].Atoms.Add(a);

            }
        }

        void testNeigbor()
        {
            int i = 0;
            float3[] positions = new float3[1000];
            float[] r = new float[1000];
            for (int x = 0; x < 10; x++)
            {
                for (int y = 0; y < 10; y++)
                {
                    for (int z = 0; z < 10; z++)
                    {
                        positions[i] = new float3(x, y, z);
                        r[i] = 0.5f;
                        i++;
                    }
                }
            }
            AtomGraph graph = new AtomGraph(positions, r, new int3(10));
            graph.SetupNeighbors();
        }
    }

}