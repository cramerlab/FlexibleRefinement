using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ProjClassifier.Tools;
using Warp;
using Warp.Tools;

namespace FlexibleRefinement.Util
{
    public class Simulation
    {

        public static float3 equidistantSpacing(float3 start, float3 end, int i, int n)
        {
            return (end - start) / (n - 1) * i + start;
        }

        public static float3 equidistantArcSpacing(float3 center, float3 a, float3 b, float R, float start, float end, int i, int n)
        {
            float angle = (end - start) / (n - 1) * i + start;
            return new float3((float)(center.X + R * Math.Cos(angle) * a.X + R * Math.Sin(angle) * b.X),
                (float)(center.Y + R * Math.Cos(angle) * a.Y + R * Math.Sin(angle) * b.Y),
                (float)(center.Z + R * Math.Cos(angle) * a.Z + R * Math.Sin(angle) * b.Z));
        }


        public static void createArc(float angle, String name)
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

        public static void simulateRotatedStick(float c)
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

        public static void simulateRotatedStick(float c, String outdir, String prefix, Action<Image> processImage = null)
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


        private static void AddText(FileStream fs, string value)
        {
            byte[] info = new UTF8Encoding(true).GetBytes(value);
            fs.Write(info, 0, info.Length);
        }

        public static double evalGauss(double x, double sigma)
        {
            return evalGauss2(Math.Pow(x, 2), Math.Pow(sigma, 2));
        }

        public static double evalGauss2(double xSqrd,double sigmaSqrd)
        {
            return Math.Exp(- xSqrd / sigmaSqrd);
        }

        public static void simulatePulledApo()
        {
            String outdir = $@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\empiar_10216\ds4";
            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }
            
            Image reference = Image.FromFile(@"D:\ferritin\empiar_10216\ref.mrc");
            Image refCropped = new Image(new int3(260, 260, 260));
            int3 dim = reference.Dims;

            Image mask = Image.FromFile(@"D:\ferritin\empiar_10216\mask.mrc");
            Image maskCropped = new Image(new int3(260, 260, 260));
            reference.Multiply(mask);

            reference.WriteMRC($@"{outdir}\ref.mrc");
            mask.WriteMRC($@"{outdir}\mask.mrc");


            float[][] refData = reference.GetHost(Intent.Read);
            refCropped.TransformValues((x, y, z, value) => {
                return refData[z + 50][(y + 50) * dim.X + (x + 50)];
            });
            refCropped.WriteMRC($@"{outdir}\refCropped.mrc");


            float[][] maskData = mask.GetHost(Intent.Read);
            maskCropped.TransformValues((x, y, z, value) => {
                return maskData[z + 50][(y + 50) * dim.X + (x + 50)];
            });
            maskCropped.WriteMRC($@"{outdir}\maskCropped.mrc");

            Image refDownsampled = ImageProcessor.Downsample(refCropped, 4);
            GPU.Normalize(refDownsampled.GetDevice(Intent.ReadWrite), refDownsampled.GetDevice(Intent.ReadWrite), (uint)refDownsampled.Dims.Elements(), 1u);
            refDownsampled.WriteMRC($@"{outdir}\refDownsampled.mrc");

            Image maskDownsampled = ImageProcessor.Downsample(maskCropped, 4);
            maskDownsampled.Binarize(1.0f);
            maskDownsampled.WriteMRC($@"{outdir}\maskDownsampled.mrc");
            int voxelSum = (int)Math.Round(maskDownsampled.AsSum3D().GetHost(Intent.Read)[0][0], 0);


            AtomGraph graph = new AtomGraph(refDownsampled, maskDownsampled, voxelSum);
            
            graph.save($@"{outdir}\startGraph.xyz");
            AtomGraph startGraph = new AtomGraph($@"{outdir}\startGraph.xyz", refDownsampled);

            Image startIm = startGraph.Repr(1.0d * graph.R0);
            startIm.WriteMRC($@"{outdir}\startIm_fromGraph.mrc");
            startIm.Binarize((float)(1.0f / Math.E));
            startIm.WriteMRC($@"{outdir}\startMask_fromGraph.mrc");
            
            float3[][] forceField  = Helper.ArrayOfFunction(i => new float3[refDownsampled.Dims.X * refDownsampled.Dims.Y], refDownsampled.Dims.Z);

            for (int z = 0; z < refDownsampled.Dims.Z; z++)
            {
                for (int y = 0; y < refDownsampled.Dims.Y; y++)
                {
                    for (int x = 0; x < refDownsampled.Dims.X; x++)
                    {
                        forceField[z][y * refDownsampled.Dims.X + x] = new float3(0);
                        //if (x > refDownsampled.Dims.X/2.0f && y > refDownsampled.Dims.Y/2.0f && z > refDownsampled.Dims.Z/2.0f)
                        //{
                        float3 dir = new float3(x - refDownsampled.Dims.X / 2.0f, y - refDownsampled.Dims.Y / 2.0f, z - refDownsampled.Dims.Z / 2.0f);
                        float3 sphericDir = new float3(dir.Length(), (float)Math.Acos(dir.Z / dir.Length()), (float)Math.Atan2(dir.Y, dir.X));
                        if (float.IsNaN(sphericDir.Y))
                            sphericDir.Y = 0;
                        if (float.IsNaN(sphericDir.Z))
                            sphericDir.Z = 0;
                        double r = dir.Length();
                        if (r > 0.0)
                        {
                            forceField[z][y * refDownsampled.Dims.X + x] = dir.Normalized() * (float)(Math.Round(evalGauss(sphericDir.X - 30, 30) * evalGauss(sphericDir.Y - Math.PI / 4, Math.PI / 4) * evalGauss(sphericDir.Z - Math.PI / 4, Math.PI / 4), 4));
                        }
                        //}

                    }
                }
            }

            int it = 100;
            for (int i = 0; i < it; i++)
            {
                graph.moveAtoms(forceField);
            }


            using (FileStream fs = File.Create($@"{outdir}\gtDisplacements.txt"))
            {
                for (int i = 0; i < graph.Atoms.Count; i++)
                {
                    AddText(fs, $"{graph.Atoms[i].Pos.X - startGraph.Atoms[i].Pos.X} {graph.Atoms[i].Pos.Y - startGraph.Atoms[i].Pos.Y} {graph.Atoms[i].Pos.Z - startGraph.Atoms[i].Pos.Z}\n".Replace(',', '.'));
                }
            }
           

            graph.save($@"{outdir}\TargetGraph{it}.xyz");
            Image tarIm = graph.Repr(1.0d * graph.R0);
            tarIm.WriteMRC($@"{outdir}\TargetIm_fromGraph{it}.mrc");
            tarIm.Binarize((float)(1.0f / Math.E));
            tarIm.WriteMRC($@"{outdir}\TargetMask_fromGraph{it}.mrc");
        }

        public static void simulatePulledToy()
        {
            String outdir = $@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy\100\";
            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }
            int3 Dims = new int3(100);
            Image startIm = new Image(Dims);
            float[][] startImData = startIm.GetHost(Intent.Write);

            double r = 4;
            double R = 10;
            int3[] mids = new int3[3] { new int3(30, 50, 50), new int3(50, 50, 50), new int3(70, 50, 50) };
            double phi = 0.0;
            double theta = 0.0;
            foreach (var mid in mids)
            {
                for (r = 0; r < 4; r += 0.1)
                {
                    for (phi = 0; phi <= 2 * Math.PI; phi += 0.1)
                    {
                        for (theta = 0; theta <= 2 * Math.PI; theta += 0.1)
                        {
                            float3 vecR = new float3((float)((R + r * Math.Cos(phi)) * Math.Cos(theta)), (float)((R + r * Math.Cos(phi)) * Math.Sin(theta)), (float)(r * Math.Sin(phi)));
                            int3 pos = new int3(mid.X + (int)Math.Round(vecR.X, 0), mid.Y + (int)Math.Round(vecR.Y, 0), mid.Z + (int)Math.Round(vecR.Z, 0));
                            startImData[pos.Y][pos.Z * Dims.X + pos.X] = 1.0f;
                        }
                    }
                }
            }
            startIm = startIm.AsConvolvedGaussian(1.0f);
            startIm.WriteMRC($@"{outdir}\startIm.mrc");
            Image startMask = startIm.GetCopy();
            startMask.Binarize((float)(1 / Math.E));
            startMask.WriteMRC($@"{outdir}\startMask.mrc");

            int nonZero = (int)(Math.Round(startMask.AsSum3D().GetHost(Intent.Read)[0][0],0));

            float3[][] forceField = Helper.ArrayOfFunction(i => new float3[Dims.X * Dims.Y],Dims.Z);
            Image forceImX = new Image(Dims);
            Image forceImY = new Image(Dims);
            Image forceImZ = new Image(Dims);
            float[][] forceImXData = forceImX.GetHost(Intent.Write);
            float[][] forceImYData = forceImY.GetHost(Intent.Write);
            float[][] forceImZData = forceImZ.GetHost(Intent.Write);
            for (int z = 0; z < Dims.Z; z++)
            {
                for (int y = 0; y < Dims.Y; y++)
                {
                    for (int x = 0; x < Dims.X; x++)
                    {
                        forceField[z][y * Dims.X + x] = new float3(0);
                      
                        //if (x > refDownsampled.Dims.X/2.0f && y > refDownsampled.Dims.Y/2.0f && z > refDownsampled.Dims.Z/2.0f)
                        //{
                        float3 dir = new float3(x - Dims.X / 2.0f, y - Dims.Y / 2.0f, z - Dims.Z / 2.0f);
                        if (dir.Y < 0 && dir.X > 0)
                        {
                            ;
                        }
                        //order: \rho, \theta, \phi
                        dir.Z = 0.0f;
                        float3 sphericDir = new float3(dir.Length(), (float)Math.Acos(dir.Z / dir.Length()), (float)Math.Atan2(dir.Y, dir.X));
                        if (float.IsNaN(sphericDir.Y))
                            sphericDir.Y = 0;
                        if (float.IsNaN(sphericDir.Z))
                            sphericDir.Z = 0;
                        r = dir.Length();
                        if (r > 0.0 && Math.Abs(sphericDir.Z) <= Math.PI/2)
                        {
                            sphericDir.Z = (float)(sphericDir.Z + Math.PI / 2);
                            sphericDir.Y = (float)(Math.PI/2);
                            float tmpX = (float)(sphericDir.X * Math.Sin(sphericDir.Y) * Math.Cos(sphericDir.Z));
                            float tmpY = (float)(sphericDir.X * Math.Sin(sphericDir.Y) * Math.Sin(sphericDir.Z));
                            float tmpZ = 0;//(float)(sphericDir.X * Math.Cos(sphericDir.Y));
                            forceField[z][y * Dims.X + x] = (new float3(tmpX, tmpY, tmpZ)) * 0.5f;
                            forceImXData[z][y * Dims.X + x] = forceField[z][y * Dims.X + x].X;
                            forceImYData[z][y * Dims.X + x] = forceField[z][y * Dims.X + x].Y;
                            forceImZData[z][y * Dims.X + x] = forceField[z][y * Dims.X + x].Z;
                        }
                        
                        //}

                    }
                }
            }
            forceImX.WriteMRC($@"{outdir}\forceImX.mrc");
            forceImY.WriteMRC($@"{outdir}\forceImY.mrc");
            forceImZ.WriteMRC($@"{outdir}\forceImZ.mrc");
            AtomGraph startGraph = new AtomGraph(startIm, startMask, nonZero);
            startGraph.save($@"{outdir}\StartGraph.xyz");
            startGraph.Repr(1.0f*startGraph.R0).WriteMRC($@"{outdir}\StartIm_fromGraph.mrc");

            AtomGraph targetGraph = new AtomGraph($@"{outdir}\StartGraph.xyz", startIm);
            
            int it = 100;
            for (int i = 0; i < it; i++)
            {
                targetGraph.moveAtoms(forceField);
            }


            using (FileStream fs = File.Create($@"{outdir}\gtDisplacements.txt"))
            {
                for (int i = 0; i < startGraph.Atoms.Count; i++)
                {
                    AddText(fs, $"{targetGraph.Atoms[i].Pos.X - startGraph.Atoms[i].Pos.X} {targetGraph.Atoms[i].Pos.Y - startGraph.Atoms[i].Pos.Y} {targetGraph.Atoms[i].Pos.Z - startGraph.Atoms[i].Pos.Z}\n".Replace(',', '.'));
                }
            }


            targetGraph.save($@"{outdir}\TargetGraph{it}.xyz");
            Image tarIm = targetGraph.Repr(1.0d * targetGraph.R0);
            tarIm.WriteMRC($@"{outdir}\TargetIm_fromGraph{it}.mrc");
            tarIm.Binarize((float)(1.0f / Math.E));
            tarIm.WriteMRC($@"{outdir}\TargetMask_fromGraph{it}.mrc");
        }
    }
}
