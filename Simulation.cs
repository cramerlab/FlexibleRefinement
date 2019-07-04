using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Warp;
using Warp.Tools;

namespace FlexibleRefinement
{
    class Simulation
    {
        static float3 equidistantSpacing(float3 start, float3 end, int i, int n)
        {
            return (end - start) / (n - 1) * i + start;
        }

        static float3 equidistantArcSpacing(float3 center, float3 a, float3 b, float R, float start, float end, int i, int n)
        {
            float angle = (end - start) / (n - 1) * i + start;
            return new float3((float)(center.X + R * Math.Cos(angle) * a.X + R * Math.Sin(angle) * b.X), 
                (float)(center.Y + R * Math.Cos(angle) * a.Y + R * Math.Sin(angle) * b.Y), 
                (float)(center.Z + R * Math.Cos(angle) * a.Z + R * Math.Sin(angle) * b.Z));
        }



        static void simulate()
        {
            int3 dims = new int3(200, 200, 200);
            Image volStick = new Image(dims);
            Image volArc = new Image(dims);
            Image volAtomsStick = new Image(dims);
            Image volAtomsArc = new Image(dims);
            float3[] atomPositionsStick = new float3[10];
            float3[] atomPositionsArc = new float3[10];
            float r = 20f;
            float rS = (float)Math.Pow(r,2);
            int n = 10;
            float len = 125;
            (float3 start, float3 end) beadString = ( new float3(dims.X/2-len/2, dims.Y/2, dims.Z/2), new float3(dims.X / 2+ len/2, dims.Y / 2, dims.Z / 2));
            float R = (float)(len/Math.PI);
            float3 c = new float3(dims.X/2, dims.Y/2, dims.Z/2);
            float3 b = new float3(1, 0, 0);
            float3 a = new float3(0, 1, 0);
            for (int i = 0; i < n; i++)
            {
                atomPositionsStick[i] = equidistantSpacing(new float3(dims.X / 2 - len / 2, dims.Y / 2, dims.Z / 2), new float3(dims.X / 2 + len / 2, dims.Y / 2, dims.Z / 2), i, n);
                atomPositionsArc[i] = equidistantArcSpacing(c, a, b, R, (float)(-Math.PI / 2), (float)(Math.PI / 2), i, n);
            }
            float[][] volStickData = volStick.GetHost(Intent.Write);
            float[][] volArcData = volArc.GetHost(Intent.Write);
            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 20, null, (z,id,ts) =>

            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {

                        for (int i = 0; i < n; i++)
                        {
                            double distStick = Math.Pow(atomPositionsStick[i].X - x, 2) + Math.Pow(atomPositionsStick[i].Y - y, 2) + Math.Pow(atomPositionsStick[i].Z - z, 2);
                            volStickData[z][dims.X * y + x] += (float)Math.Exp(-distStick / rS);
                            double distArc = Math.Pow(atomPositionsArc[i].X - x, 2) + Math.Pow(atomPositionsArc[i].Y - y, 2) + Math.Pow(atomPositionsArc[i].Z - z, 2);
                            volArcData[z][dims.X * y + x] += (float)Math.Exp(-distArc / rS);
                        }

                    }
                }
            }
             , null);
            volStick.Binarize((float)(1 / Math.E));
            volStick.WriteMRC("StickVolume_Created.mrc");
            float RAtomStick, RAtomArc;
            float3[] atomsStick = PhysicsHelper.FillWithEquidistantPoints(volStick, 1000, out RAtomStick);

            volArc.Binarize((float)(1 / Math.E));
            volArc.WriteMRC("ArcVolume_Created.mrc");
            float3[] atomsArc = PhysicsHelper.FillWithEquidistantPoints(volArc, 1000, out RAtomArc);
            float RAtomStickS = (float)Math.Pow(RAtomStick,2), RAtomArcS = (float)Math.Pow(RAtomArc/2,2);
            volStickData = volAtomsStick.GetHost(Intent.Write);
            volArcData = volAtomsArc.GetHost(Intent.Write);
            Helper.ForCPU(0, dims.Z, 20, null, (z, id, ts) =>
            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {

                        for (int i = 0; i < atomsArc.Length; i++)
                        {
                            
                            double distArc = Math.Pow(atomsArc[i].X - x, 2) + Math.Pow(atomsArc[i].Y - y, 2) + Math.Pow(atomsArc[i].Z - z, 2);
                            volArcData[z][dims.X * y + x] += distArc < RAtomArc ? 1 : 0;
                            //volArcData[z][dims.X * y + x] += (float)Math.Exp(-distArc/RAtomArcS);
                        }

                        for (int i = 0; i < atomsStick.Length; i++)
                        {
                            double distStick = Math.Pow(atomsStick[i].X - x, 2) + Math.Pow(atomsStick[i].Y - y, 2) + Math.Pow(atomsStick[i].Z - z, 2);
                            volStickData[z][dims.X * y + x] += distStick < RAtomStick ? 1 : 0;
                            //volStickData[z][dims.X * y + x] += (float)Math.Exp(-distStick / RAtomStickS);
                        }

                    }
                }
            }, null);
            volAtomsStick.WriteMRC("StickVolume_Atoms.mrc");
            volAtomsArc.WriteMRC("ArcVolume_Atoms.mrc");

            AtomGraph graph = new AtomGraph(atomsStick, Helper.ArrayOfFunction(i => RAtomStick, atomsStick.Length), dims);
        }

        static void Main(string[] args)
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
