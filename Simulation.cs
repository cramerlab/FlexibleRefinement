using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
            int3 dims = new int3(100, 100, 100);
            Image volStick = new Image(dims);

            Image volArc = new Image(dims);

            Image volAtomsStick = new Image(dims);
            Image volAtomsArc = new Image(dims);

            float r = 10f;
            float rS = (float)Math.Pow(r,2);
            int n = 5;
            float len = 62;
            float3[] atomPositionsStick = new float3[n];
            float3[] atomPositionsArc = new float3[n];
            (float3 start, float3 end) beadString = ( new float3(dims.X/2-len/2, dims.Y/2, dims.Z/2), new float3(dims.X / 2+ len/2, dims.Y / 2, dims.Z / 2));
            float arcAngle = (float)(Math.PI / 6);
            float R = (float)(len/arcAngle);
            float3 c = new float3((float)(dims.X/2 ), (float)(dims.Y/2 - (R-R * (1 - Math.Cos(arcAngle / 2)))), dims.Z/2);
            float3 b = new float3(1, 0, 0);
            float3 a = new float3(0, 1, 0);
            for (int i = 0; i < n; i++)
            {
                atomPositionsStick[i] = equidistantSpacing(beadString.start, beadString.end, i, n);
                atomPositionsArc[i] = equidistantArcSpacing(c, a, b, R, (float)(-arcAngle/2), (float)(arcAngle/2), i, n);
            }
            float[][] volStickData = volStick.GetHost(Intent.Write);


            float[][] volArcData = volArc.GetHost(Intent.Write);

            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z,id,ts) =>

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

            volStick.WriteMRC("StickVolume_Created.mrc");

            float3[][] gradStick = ImageProcessor.getGradient(volStick);
            float[][] gradDataStick = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (gradStick[z][i].Length()), dims.X * dims.Y), dims.Z);
            Image gradImStick = new Image(gradDataStick, dims);
            gradImStick.WriteMRC("StickGrad_Created.mrc");
            
            Image maskStick = volStick.GetCopy();
            maskStick.Binarize((float)(1 / Math.E));
            maskStick.WriteMRC("StickMask_Created.mrc");

            float RAtomStick, RAtomArc;
            float3[] atomsStick = PhysicsHelper.FillWithEquidistantPoints(maskStick, 1000, out RAtomStick);

            volArc.WriteMRC("ArcVolume_Created.mrc");

            float3[][] gradArc = ImageProcessor.getGradient(volArc);
            float[][] gradDataArc = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (gradArc[z][i].Length()), dims.X * dims.Y), dims.Z);
            Image gradImArc = new Image(gradDataArc, dims);
            gradImArc.WriteMRC("ArcGrad_Created.mrc");

            Image maskArc = volArc.GetCopy();
            maskArc.Binarize((float)(1 / Math.E));
            maskArc.WriteMRC("ArcMask_Created.mrc");

            float3[] atomsArc = PhysicsHelper.FillWithEquidistantPoints(maskArc, 1000, out RAtomArc);
            float RAtomStickS = (float)Math.Pow(RAtomStick,2), RAtomArcS = (float)Math.Pow(RAtomArc/2,2);
            volStickData = volAtomsStick.GetHost(Intent.Write);
            volArcData = volAtomsArc.GetHost(Intent.Write);
            /*
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
            */


            AtomGraph graph = new AtomGraph(atomsArc, Helper.ArrayOfFunction(i => RAtomStick, atomsArc.Length), dims);
            float3[][] forces = graph.CalculateForces(gradArc);
            float[][] normForce = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (forces[z][i].Length()), dims.X * dims.Y), dims.Z);
            
            Image imForce = new Image(normForce, dims);
            imForce.WriteMRC("Arc_to_Arc_normForce_it0.mrc");
            /*
            float[][] forceX = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (forces[z][i].X), dims.X * dims.Y), dims.Z);
            float[][] forceY = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (forces[z][i].Y), dims.X * dims.Y), dims.Z);
            float[][] forceZ = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (forces[z][i].Z), dims.X * dims.Y), dims.Z);
            imForce = new Image(forceX, dims);
            imForce.WriteMRC("Arc_to_Arc_forceX_it0.mrc");
            imForce = new Image(forceY, dims);
            imForce.WriteMRC("Arc_to_Arc_forceY_it0.mrc");
            imForce = new Image(forceZ, dims);
            imForce.WriteMRC("Arc_to_Arc_forceZ_it0.mrc");
            */

            /*
            graph.repr().WriteMRC("Arc_to_Arc_im_it0.mrc");
            for (int i = 0; i < 10; i++)
            {
                graph.moveAtoms(forces, 1.0f);
                graph.repr().WriteMRC($"Arc_to_Arc_im_it{i+1}.mrc");
            }
            */
            /*
            graph = new AtomGraph(atomsStick, Helper.ArrayOfFunction(i => RAtomStick, atomsStick.Length), dims);
            forces = graph.CalculateForces(gradStick);
            normForce = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (forces[z][i].Length()), dims.X * dims.Y), dims.Z);
            imForce = new Image(normForce, dims);
            imForce.WriteMRC("Stick_to_Stick_force_it0.mrc");
            graph.repr().WriteMRC("Stick_to_Stick_im_it0.mrc");
            for (int i = 0; i < 10; i++)
            {
                graph.moveAtoms(forces, 1.0f);
                graph.repr().WriteMRC($"Stick_to_Stick_im_it{i+1}.mrc");
            }
            */

            graph = new AtomGraph(atomsStick, Helper.ArrayOfFunction(i => RAtomStick, atomsStick.Length), dims);
            graph.SetupNeighbors();
            forces = graph.CalculateForces(gradArc);
            normForce = Helper.ArrayOfFunction(z => Helper.ArrayOfFunction(i => (forces[z][i].Length()), dims.X * dims.Y), dims.Z);
            imForce = new Image(normForce, dims);
            imForce.WriteMRC("Stick_to_Arc_force_it0.mrc");
            graph.repr().WriteMRC("Stick_to_Arc_im_it0.mrc");
            for (int i = 0; i < 5; i++)
            {
                graph.moveAtoms(forces, 1.0f);
                graph.repr().WriteMRC($"Stick_to_Arc_im_it{i+1}.mrc");
            }


        }



        static void Main(string[] args)
        {
            Console.WriteLine(Directory.GetCurrentDirectory());
            /*simulate();*/
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image arcIm = Image.FromFile("ArcVolume_Created.mrc");
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph stickGraph = new AtomGraph(stickIm, stickMask);
            
            stickGraph.repr().WriteMRC("StickGraph_Created.mrc");
            stickGraph.intRepr().WriteMRC("ArcVolume_splineRep.mrc");

            String trial = "Stick_to_Arc";

            stickGraph.repr().WriteMRC($"{trial}_im_it0.mrc");
            for (int i = 0; i < 10; i++)
            {
                stickGraph.moveAtoms();
                stickGraph.repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }

        }
    }
}
