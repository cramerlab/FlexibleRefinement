using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using Warp;
using Warp.Tools;
using ProjClassifier.Tools;

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

        static void generatePoint()
        {
            int3 dims = new int3(100, 100, 100);
            Image volPoint = new Image(dims);

            int n = 1;
            float r = 2f;
            float rS = (float)Math.Pow(r, 2);
            float len = 62;
            float3[] atomPositionsPoint = new float3[1] { new float3(dims.X / 2 - 5, dims.Y / 2 - 5, dims.Z / 2 - 5) };


            float[][] volPointkData = volPoint.GetHost(Intent.Write);


            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>

            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double distPoint = Math.Pow(atomPositionsPoint[i].X - x, 2) + Math.Pow(atomPositionsPoint[i].Y - y, 2) + Math.Pow(atomPositionsPoint[i].Z - z, 2);
                            volPointkData[z][dims.X * y + x] += (float)Math.Exp(-distPoint / rS);
                        }
                    }
                }
            }
             , null);

            volPoint.WriteMRC("PointVolumeShifted_Created.mrc");
        }


        static void generate2Points(float3 center, String name)
        {
            int3 dims = new int3(100, 100, 100);
            Image volPoint = new Image(dims);

            int n = 2;
            float r = 2f;
            float rS = (float)Math.Pow(r, 2);
            float len = 62;
            float3[] atomPositionsPoint = new float3[2] { new float3(center.X - 5, center.Y, center.Z), new float3(center.X + 5, center.Y, center.Z) };


            float[][] volPointkData = volPoint.GetHost(Intent.Write);


            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>

            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double distPoint = Math.Pow(atomPositionsPoint[i].X - x, 2) + Math.Pow(atomPositionsPoint[i].Y - y, 2) + Math.Pow(atomPositionsPoint[i].Z - z, 2);
                            volPointkData[z][dims.X * y + x] += (float)Math.Exp(-distPoint / rS);
                        }
                    }
                }
            }
             , null);

            volPoint.WriteMRC(name);
        }


        static void createArc(float angle, String name)
        {
            int3 dims = new int3(100, 100, 100);

            Image volArc = new Image(dims);

            Image volAtomsArc = new Image(dims);

            float r = 10f;
            float rS = (float)Math.Pow(r, 2);
            int n = 5;
            float len = 62;
            float3[] atomPositionsArc = new float3[n];
            (float3 start, float3 end) beadString = (new float3(dims.X / 2 - len / 2, dims.Y / 2, dims.Z / 2), new float3(dims.X / 2 + len / 2, dims.Y / 2, dims.Z / 2));
            float arcAngle = (float)(angle);
            float R = (float)(len / arcAngle);
            float3 c = new float3((float)(dims.X / 2), (float)(dims.Y / 2 - (R - R * (1 - Math.Cos(arcAngle / 2)))), dims.Z / 2);
            float3 b = new float3(1, 0, 0);
            float3 a = new float3(0, 1, 0);
            for (int i = 0; i < n; i++)
            {
                atomPositionsArc[i] = equidistantArcSpacing(c, a, b, R, (float)(-arcAngle / 2), (float)(arcAngle / 2), i, n);
            }


            float[][] volArcData = volArc.GetHost(Intent.Write);

            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>

            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {
                        for (int i = 0; i < n; i++)
                        {

                            double distArc = Math.Pow(atomPositionsArc[i].X - x, 2) + Math.Pow(atomPositionsArc[i].Y - y, 2) + Math.Pow(atomPositionsArc[i].Z - z, 2);
                            volArcData[z][dims.X * y + x] += (float)Math.Exp(-distArc / rS);
                        }
                    }
                }
            }
             , null);

            float RAtomArc;

            volArc.WriteMRC($"{name}Volume_Created.mrc");


            Image maskArc = volArc.GetCopy();
            maskArc.Binarize((float)(1 / Math.E));
            maskArc.WriteMRC($"{name}Mask_Created.mrc");
        }

        static void simulateRotated(float c)
        {
            int3 dims = new int3(100, 100, 100);
            Image volStick = new Image(dims);

            Image volArc = new Image(dims);

            Image volAtomsStick = new Image(dims);
            Image volAtomsArc = new Image(dims);

            float r = 10f;
            float rS = (float)Math.Pow(r, 2);
            int n = 5;
            float len = 62;
            float3[] atomPositionsStick = new float3[n];
            float3[] atomPositionsArc = new float3[n];
            (float3 start, float3 end) beadString = (new float3(dims.X / 2 - len / 2, dims.Y / 2, dims.Z / 2), new float3(dims.X / 2 + len / 2, dims.Y / 2, dims.Z / 2));
            float arcAngle = (float)(Math.PI / 6);
            float R = (float)(len / arcAngle);
            for (int i = 0; i < n; i++)
            {
                atomPositionsStick[i] = equidistantSpacing(beadString.start, beadString.end, i, n);
            }

            float3 com = new float3(0);
            foreach (var atom in atomPositionsStick)
            {
                com = com + atom;
            }

            com = com / atomPositionsStick.Length;

            for (int i = 0; i < atomPositionsStick.Length; i++)
            {
                atomPositionsStick[i] = com + Helpers.rotate_euler(atomPositionsStick[i] - com, new float3((float)(Math.PI / c), 0, 0));
            }
    

            float[][] volStickData = volStick.GetHost(Intent.Write);

          
            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>
            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double distStick = Math.Pow(atomPositionsStick[i].X - x, 2) + Math.Pow(atomPositionsStick[i].Y - y, 2) + Math.Pow(atomPositionsStick[i].Z - z, 2);
                            volStickData[z][dims.X * y + x] += (float)Math.Exp(-distStick / rS);
                        }
                    }
                }
            }
             , null);

            volStick.WriteMRC($"StickVolume_Rotated_{c}_Created.mrc");
            Image maskStick = volStick.GetCopy();
            maskStick.Binarize((float)(1 / Math.E));
            maskStick.WriteMRC($"StickMask_Rotated_{c}_Created.mrc");

            return;
        }




        static void bombard(Image target, int numShots, float width = 5.0f)
        {
            /*
             *        o x1
             *        |
             *        | d = min connection of x1 to line
             *        |
             *  o-------------o  line
             * x0             x2
             */
            Random random = new Random();
            double widthSqrd = Math.Pow(width/2, 2);

            float3 x0 = new float3(0, 0, 0);

            for (int i = 0; i < numShots; i++)
            {
                float3 randomDir = new float3((float)random.NextDouble(), (float)random.NextDouble(), (float)random.NextDouble());
                float3 x2 = x0 + randomDir;

                target.TransformValues((x, y, z, value) => {
                    float3 x1 = new float3(x, y, z);
                    float3 A = (x0 - x1);
                    float3 B = (x0 - x2);
                    float3 C = (x2 - x1);
                    float3 AxB = new float3(A.Y * B.Z - A.Z * B.Y, A.Z * B.X - A.X - B.Z, A.X * B.Y - A.Y - B.Z); 
                    double d = Math.Abs(AxB.Length()/(x2-x1).Length());
                    if (d <= width/2)
                        return 0.0f;
                    else
                        return value;
                });
            }
        }

        static void middleHole(Image target, float width = 5.0f)
        {
            /*
             *        o x1
             *        |
             *        | d = min connection of x1 to line
             *        |
             *  o-------------o  line
             * x0             x2
             */

            float3 x1 = new float3(target.Dims.X / 2, target.Dims.Y / 2, target.Dims.Z / 2);

            float3 x2 = x1 + new float3(0, 1, 0);
            float3 A = (x2 - x1);
            float3 C = (x2 - x1);
            target.TransformValues((x, y, z, value) =>
            {
                float3 x0 = new float3(x, y, z);
                
                float3 B = (x1 - x0);
                
                float3 AxB = new float3(A.Y * B.Z - A.Z * B.Y, A.Z * B.X - A.X * B.Z, A.X * B.Y - A.Y - B.X);
                double d = Math.Abs(AxB.Length() / (x2 - x1).Length());
                if (d <= width / 2)
                    return 0.0f;
                else
                    return value;
            });

        }

        static void simulateRotated(float c, String outdir, String prefix, Action<Image> processImage=null)
        {
            int3 dims = new int3(100, 100, 100);
            Image volume = new Image(dims);


            float r = 10f;
            float rS = (float)Math.Pow(r, 2);
            int n = 5;
            float len = 62;
            float3[] atomPositionsStick = new float3[n];
            float3[] atomPositionsArc = new float3[n];
            (float3 start, float3 end) beadString = (new float3(dims.X / 2 - len / 2, dims.Y / 2, dims.Z / 2), new float3(dims.X / 2 + len / 2, dims.Y / 2, dims.Z / 2));
            float arcAngle = (float)(Math.PI / 6);
            float R = (float)(len / arcAngle);
            for (int i = 0; i < n; i++)
            {
                atomPositionsStick[i] = equidistantSpacing(beadString.start, beadString.end, i, n);
            }


            float[][] volumeData = volume.GetHost(Intent.Write);


            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>
            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double distStick = Math.Pow(atomPositionsStick[i].X - x, 2) + Math.Pow(atomPositionsStick[i].Y - y, 2) + Math.Pow(atomPositionsStick[i].Z - z, 2);
                            volumeData[z][dims.X * y + x] += (float)Math.Exp(-distStick / rS);
                        }
                    }
                }
            }
             , null);

            if (processImage != null)
            {
                processImage(volume);
            }
            Image mask = volume.GetCopy();
            mask.Binarize((float)(1 / Math.E));

            if (prefix != "")
            {
                volume.WriteMRC($@"{outdir}\{prefix}_initial_volume.mrc");
                mask.WriteMRC($@"{outdir}\{prefix}_initial_mask.mrc");
            }
            else
            {
                volume.WriteMRC($@"{outdir}\initial_volume.mrc");
                mask.WriteMRC($@"{outdir}\initial_mask.mrc");
            }


            /*
            float3 com = new float3(0);
            foreach (var atom in atomPositionsStick)
            {
                com = com + atom;
            }

            com = com / atomPositionsStick.Length;

            for (int i = 0; i < atomPositionsStick.Length; i++)
            {
                atomPositionsStick[i] = com + Helpers.rotate_euler(atomPositionsStick[i] - com, new float3((float)(Math.PI / c), 0, 0));
            }

            volume.Dispose();
            volume = new Image(dims);
            volumeData = volume.GetHost(Intent.Write);


            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>
            {
                for (int y = 0; y < dims.Y; y++)
                {
                    for (int x = 0; x < dims.X; x++)
                    {
                        for (int i = 0; i < n; i++)
                        {
                            double distStick = Math.Pow(atomPositionsStick[i].X - x, 2) + Math.Pow(atomPositionsStick[i].Y - y, 2) + Math.Pow(atomPositionsStick[i].Z - z, 2);
                            volumeData[z][dims.X * y + x] += (float)Math.Exp(-distStick / rS);
                        }
                    }
                }
            }
             , null);
             */
            GPU.Rotate2D(volume.GetDevice(Intent.Read), volume.GetDevice(Intent.Write), new int2(dims.X, dims.Y), Helper.ArrayOfFunction(i => (float)(Math.PI / c), volume.Dims.Z), 1, (uint)volume.Dims.Z); 

           mask = volume.GetCopy();
           mask.Binarize((float)(1 / Math.E));

            if (prefix != "")
            {
                volume.WriteMRC($@"{outdir}\{prefix}_rotated_{c}_volume.mrc");
                mask.WriteMRC($@"{outdir}\{prefix}_rotated_{c}_mask.mrc");
            }
            else
            {
                volume.WriteMRC($@"{outdir}\rotated_{c}_volume.mrc");
                mask.WriteMRC($@"{outdir}\rotated_{c}_mask.mrc");
            }
            return;
        }



        static void simulate()
        {
            int3 dims = new int3(100, 100, 100);
            Image volStick = new Image(dims);

            Image volArc = new Image(dims);

            Image volAtomsStick = new Image(dims);
            Image volAtomsArc = new Image(dims);

            float r = 10f;
            float rS = (float)Math.Pow(r, 2);
            int n = 5;
            float len = 62;
            float3[] atomPositionsStick = new float3[n];
            float3[] atomPositionsArc = new float3[n];
            (float3 start, float3 end) beadString = (new float3(dims.X / 2 - len / 2, dims.Y / 2, dims.Z / 2), new float3(dims.X / 2 + len / 2, dims.Y / 2, dims.Z / 2));
            float arcAngle = (float)(Math.PI / 6);
            float R = (float)(len / arcAngle);
            float3 c = new float3((float)(dims.X / 2), (float)(dims.Y / 2 - (R - R * (1 - Math.Cos(arcAngle / 2)))), dims.Z / 2);
            float3 b = new float3(1, 0, 0);
            float3 a = new float3(0, 1, 0);
            for (int i = 0; i < n; i++)
            {
                atomPositionsStick[i] = equidistantSpacing(beadString.start, beadString.end, i, n);
                atomPositionsArc[i] = equidistantArcSpacing(c, a, b, R, (float)(-arcAngle / 2), (float)(arcAngle / 2), i, n);
            }
            float[][] volStickData = volStick.GetHost(Intent.Write);


            float[][] volArcData = volArc.GetHost(Intent.Write);

            //for (int z = 0; z < dims.Z; z++)
            Helper.ForCPU(0, dims.Z, 1, null, (z, id, ts) =>

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
            return;
            float3[] atomsArc = PhysicsHelper.FillWithEquidistantPoints(maskArc, 1000, out RAtomArc);
            float RAtomStickS = (float)Math.Pow(RAtomStick, 2), RAtomArcS = (float)Math.Pow(RAtomArc / 2, 2);
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
            graph.Repr().WriteMRC("Stick_to_Arc_im_it0.mrc");
            for (int i = 0; i < 5; i++)
            {
                graph.moveAtoms(forces, 1.0f);
                graph.Repr().WriteMRC($"Stick_to_Arc_im_it{i + 1}.mrc");
            }


        }


        static void shift2points()
        {
            String trial = "Shift2Points";



            /*simulate();*/

            generate2Points(new float3(50, 50, 50), "2PointVolume_Created.mrc");
            generate2Points(new float3(50, 55, 50), "2PointVolumeShifted_Created.mrc");

            /* Create Graphs */
            Image pointIm = Image.FromFile("2PointVolume_Created.mrc");
            Image pointShiftedIm = Image.FromFile("2PointVolumeShifted_Created.mrc");
            Image pointMask = pointIm.GetCopy();
            pointMask.Binarize((float)(1.0f / Math.E));
            Image pointShiftedMask = pointShiftedIm.GetCopy();
            pointShiftedMask.Binarize((float)(1.0f / Math.E));
            //AtomGraph pointGraph = new AtomGraph(pointShiftedIm, pointMask, 2);
            //AtomGraph pointShiftedGraph = new AtomGraph(pointShiftedIm, pointShiftedMask, 2);

            AtomGraph pointGraph = new AtomGraph(pointShiftedIm, pointMask, 2);
            AtomGraph pointShiftedGraph = new AtomGraph(pointShiftedIm, pointShiftedMask, 2);

            /* iterate Movements */

            int i = 0;

            pointShiftedGraph.Repr().WriteMRC($"{trial}_Target.mrc");
            pointGraph.Repr().WriteMRC($"{trial}_Start.mrc");

            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(50));
            for (; i < 20; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(35));
            for (; i < 40; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(20));
            for (; i < 60; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(10));
            for (; i < 80; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm);
            for (; i < 100; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
        }

        static void stickToArc()
        {
            String trial = "Stick_to_Arc";
            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            /* Load Images */
            //simulate();
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image arcIm = Image.FromFile("ArcVolume_Created.mrc");
            GPU.Normalize(arcIm.GetDevice(Intent.Read),
                         arcIm.GetDevice(Intent.Write),
                         (uint)arcIm.ElementsReal,
                         (uint)1);
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            Image arcMask = arcIm.GetCopy();
            arcMask.Binarize((float)(1.0f / Math.E));
            arcMask.WriteMRC($@"{trial}\arcMask.mrc");
            /* Create Graphs */
            AtomGraph startGraph = new AtomGraph(stickIm, stickMask, 1000);
            AtomGraph targetGraph = new AtomGraph(arcIm, arcMask, 1000);

            /* iterate Movements */


            targetGraph.Repr().WriteMRC($@"{trial}\{trial}_Target.mrc");
            startGraph.Repr().WriteMRC($@"{trial}\{trial}_Start.mrc");

            for (int j = 0; j < 10; j++)
            {
                int i = 0;
                arcMask = Image.FromFile($"Arc.Pi.6.{j}Mask_Created.mrc");
                startGraph.setEMIntensities(arcMask.AsConvolvedGaussian(5));
                for (; i < 5; i++)
                {
                    startGraph.moveAtoms(5.0f, 2.0f);
                    startGraph.Repr().WriteMRC($@"{trial}\{trial}_{j}_im_it{i + 1}.mrc");
                }
                startGraph.setEMIntensities(arcMask.AsConvolvedGaussian(1));
                for (; i < 10; i++)
                {
                    startGraph.moveAtoms(10.0f, 4.0f);
                    startGraph.Repr().WriteMRC($@"{trial}\{trial}_{j}_im_it{i + 1}.mrc");
                }
                startGraph.save($@"{trial}\{trial}_{j}.graph");
            }
        }

        private static void AddText(FileStream fs, string value)
        {
            byte[] info = new UTF8Encoding(true).GetBytes(value);
            fs.Write(info, 0, info.Length);
        }

        private static void EvalStickToArc()
        {

            String trial = "Stick_to_Arc";
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph initial = new AtomGraph($@"{trial}\Stick_to_Arc_initial.graph", stickIm);
            AtomGraph[] loaded = Helper.ArrayOfFunction(i => new AtomGraph($@"{trial}\Stick_to_Arc_{i}.graph", stickIm), 10);

            float3[][] displacements = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new float3(0.0f), loaded[0].Atoms.Count), 10);
            List<float>[] atomDistances = Helper.ArrayOfFunction(i => new List<float>(), 10);
            using (FileStream fs = File.Create($@"{trial}\{trial}_distanceList_initial.txt"))
            {
                for (int j = 0; j < initial.Atoms.Count; j++)
                {
                    foreach (var atom in initial.Atoms[j].Neighbours)
                    {
                        AddText(fs, $"{(atom.Pos - initial.Atoms[j].Pos).Length()}\n".Replace(",", "."));
                    }
                }
            }
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < initial.Atoms.Count; j++)
                {
                    displacements[i][j] = loaded[i].Atoms[j].Pos - initial.Atoms[j].Pos;
                    foreach (var atom in loaded[i].Atoms[j].Neighbours)
                    {
                        atomDistances[i].Add((atom.Pos - loaded[i].Atoms[j].Pos).Length());
                    }
                }
            }
            for (int i = 0; i < 10; i++)
            {
                using (FileStream fs = File.Create($@"{trial}\{trial}_distanceList_{i}.txt"))
                {
                    foreach (var d in atomDistances[i])
                    {
                        AddText(fs, $"{d}\n".Replace(',', '.'));
                    }

                }

                using (FileStream fs = File.Create($@"{trial}\{trial}_displacementList_{i}.txt"))
                {
                    foreach (var d in displacements[i])
                    {
                        AddText(fs, $"{d.X} {d.Y} {d.Z}\n".Replace(',', '.'));
                    }
                }

            }

        }

        private static void rotateGraph(String initialFile, String finalFile, String displacementFile, float c)
        {
            String trial = "RotateStick"; if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            //Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            //Image stickMask = stickIm.GetCopy();
            //stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph initial = new AtomGraph(initialFile, null);
            var atomList = initial.Atoms;
            float3 com = new float3(0);
            foreach (var atom in atomList)
            {
                com = com + atom.Pos;
            }

            com = com / atomList.Count;
            using (FileStream fs = File.Create(displacementFile))
            {

                foreach (var atom in atomList)
                {
                    float3 before = atom.Pos;
                    atom.Pos = com + Helpers.rotate_euler(atom.Pos - com, new float3((float)(Math.PI / c), 0, 0));
                    float3 after = atom.Pos;
                    float3 gtDisplacement = after - before;
                    AddText(fs, $"{gtDisplacement.X} {gtDisplacement.Y} {gtDisplacement.Z}\n".Replace(',', '.'));
                }
            }
            initial.save(finalFile);
            //initial.Repr().WriteMRC(finalFile.Replace(".graph", ".mrc"));
        }

        private static void createRotated(int c)
        {
            String trial = "RotateStick"; if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph initial = new AtomGraph($@"{trial}\Stick_Initial.graph", stickIm);
            var atomList = initial.Atoms;
            float3 com = new float3(0);
            foreach (var atom in atomList)
            {
                com = com + atom.Pos;
            }

            com = com / atomList.Count;
            using (FileStream fs = File.Create($@"{trial}\{trial}_gtDisplacementList_PI_{c}.txt"))
            {

                foreach (var atom in atomList)
                {
                    float3 before = atom.Pos;
                    atom.Pos = com + Helpers.rotate_euler(atom.Pos - com, new float3((float)(Math.PI / c), 0, 0));
                    float3 after = atom.Pos;
                    float3 gtDisplacement = after - before;
                    AddText(fs, $"{gtDisplacement.X} {gtDisplacement.Y} {gtDisplacement.Z}\n".Replace(',', '.'));
                }
            }
            initial.save($@"{trial}\Rotate_PI_{c}_gt.graph");
            //initial.Repr().WriteMRC($@"{trial}\Rotate_PI_{c}_gt.mrc");
        }



        static Image Downsample(Image im, float factor)
        {
            int3 oldSize = im.Dims;
            float falloff = 5.0f;
            int3 newSize = oldSize / factor;
            float innerRadius = (newSize-newSize/2).Length() - (1.1f * falloff);
            Image ft = im.AsFFT(true);
            Image Cosine = new Image(ft.Dims, true);
            float[][] CosineData = Cosine.GetHost(Intent.Write);
            double CosineSum = 0;
            for (int z = 0; z < Cosine.DimsFT.Z; z++)
            {
                int zz = z;
                if (z > Cosine.DimsFT.Z / 2)
                {
                    zz = Cosine.DimsFT.Z - z;
                }
                zz *= zz;
                for (int y = 0; y < Cosine.DimsFT.Y; y++)
                {
                    int yy = y;
                    if ( y > Cosine.DimsFT.Y / 2)
                    {
                        yy = Cosine.DimsFT.Y - y;
                    }
                    
                    yy *= yy;
                    for (int x = 0; x < Cosine.DimsFT.X; x++)
                    {
                        int xx = x;
                        xx *= xx;

                        float R = (float)Math.Sqrt(xx + yy + zz);
                        double C = Math.Cos(Math.Max(0, Math.Min(falloff, R - innerRadius)) / falloff * Math.PI) * 0.5 + 0.5;

                        CosineSum += C;
                        CosineData[z][y * Cosine.DimsFT.X + x] = (float)C;
                    }
                }
            }
            ft.WriteMRC("ftRemapped.mrc");
            Cosine.WriteMRC("RaisedCosineIM.mrc");
            
            ft.Multiply(Cosine);
            ft.WriteMRC("ftRemappedultiplied.mrc");
            ft = ft.AsPadded(newSize); ft.WriteMRC("ftRemappedPadded.mrc");
            Image newIm = ft.AsIFFT(true);
            GPU.Normalize(newIm.GetDevice(Intent.Read),
                         newIm.GetDevice(Intent.Write),
                         (uint)newIm.ElementsReal,
                         (uint)1);
            ft.WriteMRC("ftRemappedNormalized.mrc");
            return newIm;

        }


        public static void doRotationExp(String outdir, float c, List<float> corrScales, List<float> distScales, List<bool> normalizes)
        {
            float pixSize = 1.0f;
            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = Image.FromFile("StickMask_Created.mrc");

            if (!File.Exists($"StickVolume_Rotated_{c}_Created.mrc"))
            {
                simulateRotated(c);
            }
            Image stickRotIm = Image.FromFile($"StickVolume_Rotated_{c}_Created.mrc");
            Image stickRotMask = Image.FromFile($"StickMask_Rotated_{c}_Created.mrc");

            float[] sampleRates = new float[3] { 4, 2, 1 };
            int targetCount = 10000;
            int[] sampledCounts = Helper.ArrayOfFunction(k => (int)(targetCount / Math.Pow(sampleRates[k], 3)), 3);
            Image[] StartIms = Helper.ArrayOfFunction(k => sampleRates[k] == 1 ? stickIm.GetCopy() : Downsample(stickIm, sampleRates[k]), 3);
            Image[] StartMasks = Helper.ArrayOfFunction(k =>
            {
                Image mask = StartIms[k].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);

            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(k => new AtomGraph(StartIms[k], StartMasks[k], sampledCounts[k]), 3);


            Image[] TarIms = Helper.ArrayOfFunction(k => sampleRates[k] == 1 ? stickRotIm.GetCopy() : Downsample(stickRotIm, sampleRates[k]), 3);
            Image[] TarMasks = Helper.ArrayOfFunction(k =>
            {
                Image mask = TarIms[k].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);
            for (int k = 0; k < 3; k++)
            {
                StartIms[k].WriteMRC($@"{outdir}\{sampleRates[k]}_StartIm.mrc");
                TarIms[k].WriteMRC($@"{outdir}\{sampleRates[k]}_{c}_TarIm.mrc");

                StartMasks[k].WriteMRC($@"{outdir}\{sampleRates[k]}_StartMask.mrc");
                TarMasks[k].WriteMRC($@"{outdir}\{sampleRates[k]}_{c}_TarMask.mrc");

                StartGraphs[k].save($@"{outdir}\{sampleRates[k]}_StartGraph.graph");
                //StartGraphs[k].Repr().WriteMRC($@"{outdir}\{sampleRates[k]}_StartGraph.mrc");

                rotateGraph($@"{outdir}\{sampleRates[k]}_StartGraph.graph", $@"{outdir}\{sampleRates[k]}_{c}_TarGraph.graph", $@"{outdir}\{sampleRates[k]}_{c}_GtDisplacements", c);
                AtomGraph targetGraph = new AtomGraph($@"{outdir}\{sampleRates[k]}_{c}_TarGraph.graph", TarIms[k]);
                //targetGraph.Repr().WriteMRC($@"{outdir}\{sampleRates[k]}_{c}_TarGraph.mrc");
            }



            #region FirstStep
            if (!Directory.Exists($@"{outdir}\StepOne"))
            {
                Directory.CreateDirectory($@"{outdir}\StepOne");
            }
            TarIms[0].AsConvolvedGaussian(1).WriteMRC($@"{outdir}\StepOne\{sampleRates[0]}_{c}_TarIm_Convolved.mrc");


            DateTime begin = DateTime.UtcNow;

            int i = 0;
            AtomGraph localStartGraph = new AtomGraph($@"{outdir}\{sampleRates[0]}_StartGraph.graph", TarIms[0].AsConvolvedGaussian(1));
            for (; i < 10; i++)
            {
                localStartGraph.moveAtoms(corrScales[0], distScales[0], normalizes[0]);
                localStartGraph.save($@"{outdir}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScales[0]:#.#}_{distScales[0]:#.#}_{normalizes[0]}_it{i + 1}.graph");
            }
            localStartGraph.save($@"{outdir}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScales[0]:#.#}_{distScales[0]:#.#}_{normalizes[0]}_final.graph");

            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion


            #region SecondStep
            String fromFirstGraphFileStart = $@"{outdir}\{sampleRates[0]}_StartGraph.graph";
            AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
            String fromFirstGraphFileFinal = $@"{outdir}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScales[0]:#.#}_{distScales[0]:#.#}_{normalizes[0]}_final.graph";
            AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

            List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
            for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{outdir}\StepTwo"))
            {
                Directory.CreateDirectory($@"{outdir}\StepTwo");
            }
            begin = DateTime.UtcNow;

            i = 0;
            localStartGraph = new AtomGraph($@"{outdir}\{sampleRates[1]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
            localStartGraph.setPositions(fromFirstGraphStart, displacements);
            for (; i < 10; i++)
            {
                localStartGraph.moveAtoms(corrScales[1], distScales[1], normalizes[1]);
                localStartGraph.save($@"{outdir}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScales[1]:#.#}_{distScales[1]:#.#}_{normalizes[1]}_it{i + 1}.graph");
            }
            localStartGraph.save($@"{outdir}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScales[1]:#.#}_{distScales[1]:#.#}_{normalizes[1]}_final.graph");

            end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion



            #region ThirdStep
            String fromSecondGraphFileStart = $@"{outdir}\{sampleRates[1]}_StartGraph.graph";
            AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
            String fromSecondGraphFileFinal = $@"{outdir}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScales[1]:#.#}_{distScales[1]:#.#}_{normalizes[1]}_final.graph";
            AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

            displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
            for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{outdir}\StepThree"))
            {
                Directory.CreateDirectory($@"{outdir}\StepThree");
            }
            begin = DateTime.UtcNow;

            i = 0;
            localStartGraph = new AtomGraph($@"{outdir}\{sampleRates[2]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
            localStartGraph.setPositions(fromSecondGraphStart, displacements);
            for (; i < 10; i++)
            {
                localStartGraph.moveAtoms(corrScales[2], distScales[2], normalizes[2]);
                localStartGraph.save($@"{outdir}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScales[2]:#.#}_{distScales[2]:#.#}_{normalizes[2]}_it{i + 1}.graph");
            }
            localStartGraph.save($@"{outdir}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScales[2]:#.#}_{distScales[2]:#.#}_{normalizes[2]}_final.graph");

            end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion*/
        }

        public static void GridSearchParams(int c=10)
        {
            String trial = "GridSearchParams_No_NeighborUpdate_c10000_it100";
            float pixSize = 1.0f;

            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = Image.FromFile("StickMask_Created.mrc");

            if (!File.Exists($"StickVolume_Rotatef_{c}_Created.mrc"))
            {
                simulateRotated(c);
            }
            Image stickRotIm = Image.FromFile($"StickVolume_Rotated_{c}_Created.mrc");
            Image stickRotMask = Image.FromFile($"StickMask_Rotated_{c}_Created.mrc");

            float[] sampleRates = new float[3] { 4, 2, 1 };
            int targetCount = 10000;
            int[] sampledCounts = Helper.ArrayOfFunction(i => (int)(targetCount / Math.Pow(sampleRates[i], 3)), 3);
            Image[] StartIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? stickIm.GetCopy() : Downsample(stickIm, sampleRates[i]), 3);
            Image[] StartMasks = Helper.ArrayOfFunction(i => {
                Image mask = StartIms[i].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);

            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(i => new AtomGraph(StartIms[i], StartMasks[i], sampledCounts[i]), 3);


            Image[] TarIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? stickRotIm.GetCopy() : Downsample(stickRotIm, sampleRates[i]), 3);
            Image[] TarMasks = Helper.ArrayOfFunction(i => {
                Image mask = TarIms[i].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);
            for (int i = 0; i < 3; i++)
            {
                StartIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartIm.mrc");
                TarIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarIm.mrc");

                StartMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartMask.mrc");
                TarMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarMask.mrc");

                StartGraphs[i].save($@"{trial}\{sampleRates[i]}_StartGraph.graph");
                //StartGraphs[i].Repr().WriteMRC($@"{trial}\{sampleRates[i]}_StartGraph.mrc");

                rotateGraph($@"{trial}\{sampleRates[i]}_StartGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_GtDisplacements", c);
                AtomGraph targetGraph = new AtomGraph($@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", TarIms[i]);
               // targetGraph.Repr().WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarGraph.mrc");
            }

            float[] corrScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            bool[] normalizings = new bool[2] { true, false };
            /*
            #region FirstStep
            if (!Directory.Exists($@"{trial}\StepOne"))
            {
                Directory.CreateDirectory($@"{trial}\StepOne");
            }
            TarIms[0].AsConvolvedGaussian(1).WriteMRC($@"{trial}\StepOne\{sampleRates[0]}_{c}_TarIm_Convolved.mrc");

            
            DateTime begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) => {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[0]}_StartGraph.graph", TarIms[0].AsConvolvedGaussian(1));
                        for (;i < 100; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }

            }, null);
            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion
            */
            /*
           #region SecondStep
           String fromFirstGraphFileStart = $@"{trial}\{sampleRates[0]}_StartGraph.graph";
           AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
           String fromFirstGraphFileFinal = $@"{trial}\StepOne\4_Rotate_PI_{c}_1_10_False_final.graph";
           AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

           List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
           for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
           {
               displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
           }


           
           if (!Directory.Exists($@"{trial}\StepTwo"))
           {
               Directory.CreateDirectory($@"{trial}\StepTwo");
           }
           DateTime begin = DateTime.UtcNow;
           Helper.ForCPU(0, 20, 20, null, (k, id, ts) => {
               float corrScale = corrScales[k];

               foreach (var distScale in distScales)
               {
                   foreach (var normalizing in normalizings)
                   {
                       int i = 0;
                       AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[1]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
                       localStartGraph.setPositions(fromFirstGraphStart, displacements);
                       for (; i < 100; i++)
                       {
                           localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                           localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                       }
                       localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                   }

               }

           }, null);
           DateTime end = DateTime.UtcNow;
           System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
           #endregion
           */

            
            #region ThirdStep
            String fromSecondGraphFileStart = $@"{trial}\{sampleRates[1]}_StartGraph.graph";
           AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
           String fromSecondGraphFileFinal = $@"{trial}\StepTwo\2_Rotate_PI_{c}_7_9_False_final.graph";
           AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

           List<float3> displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
           for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
           {
               displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
           }



           if (!Directory.Exists($@"{trial}\StepThree"))
           {
               Directory.CreateDirectory($@"{trial}\StepThree");
           }
           DateTime begin = DateTime.UtcNow;
           Helper.ForCPU(0, 20, 20, null, (k, id, ts) => {
               float corrScale = corrScales[k];

               foreach (var distScale in distScales)
               {
                   foreach (var normalizing in normalizings)
                   {
                       int i = 0;
                       AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[2]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
                       localStartGraph.setPositions(fromSecondGraphStart, displacements);
                       for (; i < 50; i++)
                       {
                           localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                           localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                       }
                       localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                   }

               }

           }, null);
           DateTime end = DateTime.UtcNow;
           System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
           #endregion
          
           
            return;
        }

        public static void GridSearchParams(String startImFile, String startMaskFile, String tarImFile, String tarMaskFile, String trial, int c = 10, int targetCount = 2000)
        {
            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            int steps = 3;
            float[] sampleRates = new float[3] { 4, 2, 1 };
            int[] numIterations = new int[3] { 50, 50, 50 };
            int[] sampledCounts = Helper.ArrayOfFunction(i => (int)(targetCount / Math.Pow(sampleRates[i], 3)), 3);
            float[] corrScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            bool[] normalizings = new bool[2] { true, false };

            float[] minCorrScales = new float[steps];
            float[] minDistScales = new float[steps];
            bool[] minNormalizings = new bool[steps];


            Image startIm = Image.FromFile(startImFile);
            Image startMask = Image.FromFile(startMaskFile);
            Image tarIm = Image.FromFile(tarImFile);
            Image tarMask = Image.FromFile(tarMaskFile);



            Image[] StartIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? startIm.GetCopy() : Downsample(startIm, sampleRates[i]), steps);
            Image[] StartMasks = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? startMask.GetCopy() : Downsample(startMask, sampleRates[i]), steps);
            for (int k = 0; k < steps; k++)
            {
                StartMasks[k].Binarize(1.0f);
            }
            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(i => new AtomGraph(StartIms[i], StartMasks[i], sampledCounts[i]), steps);


            Image[] TarIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? tarIm.GetCopy() : Downsample(tarIm, sampleRates[i]), steps);
            Image[] TarMasks = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? tarMask.GetCopy() : Downsample(tarMask, sampleRates[i]), steps);
            for (int k = 0; k < steps; k++)
            {
                TarMasks[k].Binarize(1.0f);
            }


            //AtomGraph[] TargetGraphs = Helper.ArrayOfFunction(i => new AtomGraph(TarIms[i], TarMasks[i], sampledCounts[i]), steps);

            for (int i = 0; i < steps; i++)
            {
                StartIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartIm.mrc");
                TarIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarIm.mrc");

                StartMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartMask.mrc");
                TarMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarMask.mrc");

                StartGraphs[i].save($@"{trial}\{sampleRates[i]}_StartGraph.graph");

                rotateGraph($@"{trial}\{sampleRates[i]}_StartGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_GtDisplacements", c);

            }
            AtomGraph[] TargetGraphs = Helper.ArrayOfFunction(i => new AtomGraph($@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", TarIms[i]), steps);


            #region FirstStep
            if (!Directory.Exists($@"{trial}\StepOne"))
            {
                Directory.CreateDirectory($@"{trial}\StepOne");
            }
            TarIms[0].WriteMRC($@"{trial}\StepOne\{sampleRates[0]}_{c}_TarIm.mrc");


            DateTime begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) =>
            {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[0]}_StartGraph.graph", TarIms[0]);
                        for (; i < numIterations[0]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }

            }, null);
            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion

            /* Evaluate first step */

            double minDisplRMSD = double.MaxValue;

            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[0].AsConvolvedGaussian(1));
                        double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[0].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[0] = corrScale;
                            minDistScales[0] = distScale;
                            minNormalizings[0] = normalizing;
                            minDisplRMSD = displRMSD;
                        }
                    }
                }
            }
            /* */


            #region SecondStep
            String fromFirstGraphFileStart = $@"{trial}\{sampleRates[0]}_StartGraph.graph";
            AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
            String fromFirstGraphFileFinal = $@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{minCorrScales[0]}_{minDistScales[0]}_{minNormalizings[0]}_final.graph";
            AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

            List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
            for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{trial}\StepTwo"))
            {
                Directory.CreateDirectory($@"{trial}\StepTwo");
            }
            begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) =>
            {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[1]}_StartGraph.graph", TarIms[1]);
                        localStartGraph.setPositions(fromFirstGraphStart, displacements);
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.graph");
                        for (; i < numIterations[1]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                        
                    }

                }

            }, null);
            end = DateTime.UtcNow;
            System.Console.WriteLine($"Second Step Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion
            /* Evaluate second step */

            minDisplRMSD = double.MaxValue;

            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[1].AsConvolvedGaussian(1));
                        double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[1].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[1] = corrScale;
                            minDistScales[1] = distScale;
                            minNormalizings[1] = normalizing;
                        }
                    }
                }
            }
            /* */


            #region ThirdStep
            String fromSecondGraphFileStart = $@"{trial}\{sampleRates[1]}_StartGraph.graph";
            AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarMasks[1].AsConvolvedGaussian(1));
            String fromSecondGraphFileFinal = $@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{minCorrScales[1]}_{minDistScales[1]}_{minNormalizings[1]}_final.graph";
            AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarMasks[1].AsConvolvedGaussian(1));

            displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
            for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{trial}\StepThree"))
            {
                Directory.CreateDirectory($@"{trial}\StepThree");
            }
            begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) =>
            {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[2]}_StartGraph.graph", TarIms[2]);
                        localStartGraph.setPositions(fromSecondGraphStart, displacements);
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.graph");
                        for (; i < numIterations[2]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }

            }, null);
            end = DateTime.UtcNow;
            System.Console.WriteLine($"Third Step Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion
            /* Evaluate third step */

            minDisplRMSD = double.MaxValue;

            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[2].AsConvolvedGaussian(1));
                        double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[2].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[2] = corrScale;
                            minDistScales[2] = distScale;
                            minNormalizings[2] = normalizing;
                        }
                    }
                }
            }
            /* */


            return;
        }

        public static void graphTest()
        {
            String trial = "TestUpscaling"; if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            Image downsampledIm = Image.FromFile(@"downsampling\Downsampled_4.mrc");
            AtomGraph downsampled = new AtomGraph($@"downsampling\StartGraph_downSampled_4.graph", downsampledIm);

            Image nextIm = Image.FromFile(@"downsampling\Downsampled_2.mrc");
            AtomGraph nextGraph = new AtomGraph($@"downsampling\StartGraph_downSampled_2.graph", nextIm);

            downsampled.Repr().WriteMRC($@"{trial}\DownScaledGraphStart.mrc");
            rotateGraph($@"downsampling\StartGraph_downSampled_4.graph", $@"{trial}\StartGraph_downSampled_4_rotate_5.graph", $@"{trial}\StartGraph_downSampled_4_rotate_5.displacements.txt", 5);
            downsampled = new AtomGraph($@"{trial}\StartGraph_downSampled_4_rotate_5.graph", downsampledIm);
            downsampled.Repr().WriteMRC($@"{trial}\DownScaledGraphUpdated.mrc");

            AtomGraph rotated = new AtomGraph($@"{trial}\StartGraph_downSampled_4_rotate_5.graph", downsampledIm);

            List<float3> displacements = new List<float3>(rotated.Atoms.Count);
            for (int i = 0; i < rotated.Atoms.Count; i++)
            {
                displacements.Add(rotated.Atoms[i].Pos - downsampled.Atoms[i].Pos);
            }
            nextGraph.Repr().WriteMRC($@"{trial}\UpscaledGraphStart.mrc");
            nextGraph.setPositions(downsampled, displacements);
            nextGraph.Repr().WriteMRC($@"{trial}\UpscaledGraphUpdated.mrc");

        }



        static void Main(string[] args)
        {
            String rootDir = @"D:\Software\FlexibleRefinement\bin\Debug\lennardJones\bombarded";
            if (!Directory.Exists(rootDir))
            {
                Directory.CreateDirectory(rootDir);
            }
            Directory.SetCurrentDirectory(rootDir);

            //simulateRotated(8, rootDir, "", (im) => { middleHole(im, 4.0f); });
            int c = 8;
            GridSearchParams(rootDir + @"\initial_volume.mrc", rootDir + @"\initial_mask.mrc", rootDir + $@"\rotated_{c}_volume.mrc", rootDir + $@"\rotated_{c}_volume.mrc", rootDir + @"\middleHole_10000_it50_50_50", c, 10000);

            //GridSearchParams(8);
            /*for (int c = 8; c < 9; c++)
            {
                doRotationExp(@"D:\Software\FlexibleRefinement\bin\Debug\lennardJones\differentRotExp_No_NeighborUpdate_c10000", c, new List<float>(new float[] { 2, 10, 10 }), new List<float>(new float[] { 17, 13, 13 }), new List<bool>(new bool[] { false, true, true }));
            }*/
            //int c = 10;
            //createRotated(c);
            //String trial = "RotateStick"; if (!Directory.Exists(trial))
            //{
            //    Directory.CreateDirectory(trial);
            //}
            //Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            //Image stickMask = stickIm.GetCopy();
            //stickMask.Binarize((float)(1.0f / Math.E));
            //AtomGraph startGraph = new AtomGraph($@"{trial}\Stick_Initial.graph", stickIm);

            //Image rotated = Image.FromFile($@"{trial}\Rotate_PI_{c}_gt.mrc");
            //rotated.AsConvolvedGaussian(1.0f).WriteMRC($@"{trial}\Rotate_PI_{c}_gt_convolved1.mrc");
            //Image convolved4 = rotated.AsConvolvedGaussian(4.0f);
            //convolved4.WriteMRC($@"{trial}\Rotate_PI_{c}_gt_convolved4.mrc");

            //Image rotatedkMask = convolved4.GetCopy();
            //rotatedkMask.Binarize(0.25f);
            //rotatedkMask.WriteMRC($@"{trial}\Rotate_PI_{c}_gt_convolved4_mask.mrc");
            //startGraph.setEMIntensities(rotatedkMask.AsConvolvedGaussian(5));


            //float[] corrScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            //float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            //bool[] normalizings = new bool[2] { true, false };
            //Helper.ForCPU(0, 20, 11, null, (k, id, ts) => {
            //    float corrScale = corrScales[k];

            //    foreach (var distScale in distScales)
            //    {
            //        foreach (var normalizing in normalizings)
            //        {
            //            int i = 0;
            //            AtomGraph localStartGraph = new AtomGraph($@"{trial}\Stick_Initial.graph", stickIm);
            //            localStartGraph.setEMIntensities(rotatedkMask.AsConvolvedGaussian(5));
            //            for (; i < 5; i++)
            //            {
            //                localStartGraph.moveAtoms(corrScale, distScale, normalizing);
            //                localStartGraph.Repr().WriteMRC($@"{trial}\Rotate_PI_{c}_im_it{i + 1}_{corrScale:#.#}_{distScale:#.#}_{normalizing}.mrc");
            //            }
            //            localStartGraph.save($@"{trial}\Rotate_PI_{c}_final_{corrScale:#.#}_{distScale:#.#}_{normalizing}.graph");
            //        }

            //    }

            //}, null);
            //startGraph.setEMIntensities(rotatedkMask.AsConvolvedGaussian(1));

            ///*for (; i < 10; i++)
            //{
            //    startGraph.moveAtoms(10.0f, 4.0f,true);
            //    startGraph.Repr().WriteMRC($@"{trial}\Rotate_PI_{c}_im_it{i + 1}.mrc");
            //}*/
            //startGraph.save($@"{trial}\Rotate_PI_{c}_final.graph");
        }
    }
}
