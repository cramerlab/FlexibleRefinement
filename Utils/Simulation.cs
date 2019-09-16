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

        public static void simulatePulledApo()
        {
            String outdir = $@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\empiar_10216\";
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

            Image refDownsampled = ImageProcessor.Downsample(refCropped, 2);
            GPU.Normalize(refDownsampled.GetDevice(Intent.ReadWrite), refDownsampled.GetDevice(Intent.ReadWrite), (uint)refDownsampled.Dims.Elements(), 1u);
            refDownsampled.WriteMRC($@"{outdir}\refDownsampled.mrc");

            Image maskDownsampled = ImageProcessor.Downsample(maskCropped, 2);
            maskDownsampled.Binarize(1.0f);
            maskDownsampled.WriteMRC($@"{outdir}\maskDownsampled.mrc");
            AtomGraph graph = new AtomGraph(refDownsampled, maskDownsampled, 10000);
            graph.save($@"{outdir}\startGraph.xyz");
            graph.Repr(1.5d * graph.R0).WriteMRC($@"{outdir}\startGraph.mrc");
            float3[][] forceField  = Helper.ArrayOfFunction(i => new float3[refDownsampled.Dims.X * refDownsampled.Dims.Y], refDownsampled.Dims.Z);

            for (int z = 0; z < refDownsampled.Dims.Z; z++)
            {
                for (int y = 0; y < refDownsampled.Dims.Y; y++)
                {
                    for (int x = 0; x < refDownsampled.Dims.X; x++)
                    {
                        forceField[z][y * refDownsampled.Dims.X + x] = new float3(0);
                        if (x > refDownsampled.Dims.X/2.0f && y > refDownsampled.Dims.Y/2.0f && z > refDownsampled.Dims.Z/2.0f)
                        {
                            float3 dir = new float3(x - refDownsampled.Dims.X / 2.0f, y - refDownsampled.Dims.Y / 2.0f, z - refDownsampled.Dims.Z / 2.0f);
                            forceField[z][y * refDownsampled.Dims.X + x] = dir.Normalized()*(dir*0.1f).Length();

                        }
                    
                    }
                }
            }
            

            for (int i = 0; i < 50; i++)
            {
                graph.moveAtoms(forceField);
            }

            graph.save($@"{outdir}\TargetGraph.xyz");
            graph.Repr(1.5d*graph.R0).WriteMRC($@"{outdir}\TargetGraph.mrc");
        }
    }
}
