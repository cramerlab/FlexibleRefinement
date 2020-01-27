using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using FlexibleRefinement.Util;
using System.IO;
using System.Globalization;

namespace FlexibleRefinement
{
    [SuppressUnmanagedCodeSecurity]
    class Program
    {

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EntryPoint")]
        public static extern IntPtr EntryPoint(int3 dims, float[] atomCenters, float[] atomWeights, float rAtoms, uint nAtoms);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetProjection")]
        public static extern void GetProjection(IntPtr proj, float[] output, float[] output_nrm, float3 angles, float shiftX, float shiftY);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStep")]
        public static extern float DoARTStep(IntPtr proj, float[] Iexp, float3[] angles, float shiftX, float shiftY, uint numImages);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStepMoved")]
        public static extern float DoARTStepMoved(IntPtr proj, float[] Iexp, float3[] angles, float[] atomPositions, float shiftX, float shiftY, uint numImages);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetIntensities")]
        public static extern void GetIntensities(IntPtr proj, float[] outp);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "convolve")]
        public static extern void convolve(float[] img, float[] ctf, float[] outp, int3 dims);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "getGaussianTableFull")]
        public static extern void getGaussianTableFull(float[] table, float sigma, int elements);


        public static Image GetCTFCoords(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image CTFCoords;
            {
                float2[] CTFCoordsData = new float2[(size.X / 2 + 1) * size.Y];
                for (int y = 0; y < size.Y; y++)
                    for (int x = 0; x < size.X / 2 + 1; x++)
                    {
                        int xx = x;
                        int yy = y < size.Y / 2 + 1 ? y : y - size.Y;

                        float xs = xx / (float)originalSize.X;
                        float ys = yy / (float)originalSize.Y;
                        float r = (float)Math.Sqrt(xs * xs + ys * ys);
                        float angle = (float)Math.Atan2(yy, xx);

                        if (pixelSize != 1 || pixelSizeDelta != 0)
                            r /= pixelSize + pixelSizeDelta * (float)Math.Cos(2.0 * (angle - pixelSizeAngle));

                        CTFCoordsData[y * (size.X / 2 + 1) + x] = new float2(r, angle);
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1), true);
            }

            return CTFCoords;
        }


        static void Main(string[] args)
        {
            NumberFormatInfo nfi = new NumberFormatInfo();
            nfi.NumberDecimalSeparator = ".";
            Random rand = new Random();
            String outdir = @"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current";
            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }

            String indir = $@"D:\Software\FlexibleRefinement\bin\Debug\Toy_Modulated\lowHighMix\current";


            Star TableIn = new Star(@"D:\florian_debug\particles.star");
            CTF[] CTFParams = TableIn.GetRelionCTF();


            Image inputVol = Image.FromFile($@"{indir}\startIm.mrc");
            Image inputMask = Image.FromFile($@"{indir}\startMask.mrc");

            AtomGraph graph = new AtomGraph($@"{indir}\StartGraph.graph", inputVol);
            AtomGraph graphMoved = new AtomGraph($@"{indir}\TargetGraph100.graph", inputVol);
            Projector proj = new Projector(inputVol, 2);
            Projector maskProj = new Projector(inputMask, 2);


            int numParticles = 300;
            int numAngles = numParticles;
            int numAnglesX = (int)Math.Ceiling(Math.Pow(numAngles, 1.0d / 3.0d));
            int numAnglesY = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            int numAnglesZ = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            numAngles = numAnglesX * numAnglesY * numAnglesZ;
            numParticles = numAngles;

            CTFParams = CTFParams.Take(numParticles).ToArray();
            int2 particleRes = new int2(100);
            for (int i = 0; i < CTFParams.Length; i++)
                CTFParams[i].PixelSize = (decimal)1.8;// CTFParams[i].PixelSize* (decimal)2.2;

            
            for (int idx = 0; idx < CTFParams.Length; idx++)
            {
                CTFParams[idx].Defocus = (decimal)((0.4*rand.NextDouble()-0.2) + 1);
            }
            
            Image CTFCoords = GetCTFCoords(new int2(particleRes.X, particleRes.Y), new int2(particleRes.X, particleRes.Y));
            Image CTFs = new Image(new int3(particleRes.X, particleRes.Y, numParticles), true);
            GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                            CTFCoords.GetDevice(Intent.Read),
                            (uint)CTFCoords.ElementsSliceComplex,
                            CTFParams.Select(p => p.ToStruct()).ToArray(),
                            false,
                            (uint)numParticles);
            CTFs.WriteMRC($@"{outdir}\CTFS.mrc");
            int factor = 10;
            float sigma = 0.8f;

            //int2 gaussTableLayout = new int2(2 * factor * (int)Math.Ceiling(Math.Sqrt(2) * 4 * sigma), 1);
            int2 gaussTableLayout = new int2(factor * particleRes.X + 2, 1);

            Image CTFCoordsGauss = GetCTFCoords(new int2(gaussTableLayout.X, gaussTableLayout.Y), new int2(particleRes.X, particleRes.Y), 1.0f);
            Image CTFsGauss = new Image(new int3(gaussTableLayout.X, gaussTableLayout.Y, numParticles), true);
            GPU.CreateCTF(CTFsGauss.GetDevice(Intent.Write),
                            CTFCoordsGauss.GetDevice(Intent.Read),
                            (uint)CTFCoordsGauss.ElementsSliceComplex,
                            CTFParams.Select(p => p.ToStruct()).ToArray(),
                            false,
                            (uint)numParticles);

            Image Gauss = new Image(new int3(gaussTableLayout.X, gaussTableLayout.Y, numParticles));
            Gauss.TransformValues((x, y, z, v) => (float)(Math.Exp(-0.5*Math.Pow(((float)(x-gaussTableLayout.X/2)/factor) / sigma, 2))));

            Image GaussFT = Gauss.AsFFT();
            Image GaussIFT = GaussFT.AsIFFT(false, 0, true);
            GaussFT.Multiply(CTFsGauss);
            Image GaussConvolved = GaussFT.AsIFFT(false, 0, true);

            float[] gaussConvolvedFlat = Gauss.GetHostContinuousCopy();
            float[][] gaussConvolved = Helper.ArrayOfFunction(i =>
            {
                float[] inner = new float[gaussTableLayout.X / 2];
                for (int j = 0; j < gaussTableLayout.X / 2; j++)
                {
                    inner[j] = gaussConvolvedFlat[gaussTableLayout.X * i + gaussTableLayout.X/2 + j];
                }
                return inner;
            }
            , numParticles);



            float3[] angles = new float3[numAngles];
            {
                int i = 0;
                for (int x = 0; x < numAnglesX; x++)
                {
                    float xx = (float)(2 * Math.PI) / (numAnglesX - 1) * (x);
                    for (int y = 0; y < numAnglesY; y++)
                    {
                        float yy = (float)(2 * Math.PI) / (numAnglesY - 1) * (y);
                        for (int z = 0; z < numAnglesZ; z++)
                        {
                            float zz = (float)(2 * Math.PI) / (numAnglesZ - 1) * (z);
                            angles[i] = new float3(xx, yy, zz);
                            i++;
                        }
                    }
                }
            }

            int numPseudoparticles = numParticles;
            float[] projectionAngles = new float[numPseudoparticles * 3];
            for (int k = 0; k < numPseudoparticles; k++)
            {
                projectionAngles[k] = angles[k].X;
                projectionAngles[k + 1] = angles[k].Y;
                projectionAngles[k + 2] = angles[k].Z;
            }
            float[] atomPositions = graph.GetAtomPositions();
            float[] movedAtomPositions = graphMoved.GetAtomPositions();

            float[] realIntensities = graph.GetAtomIntensities();
           
            float[] startIntensities = Helper.ArrayOfFunction(i => (float)(rand.NextDouble()), graph.Atoms.Count());
            AtomGraph graphBefore = new AtomGraph(graph);
            AtomGraph targetGraph = new AtomGraph(graph);
            graphBefore.SetAtomIntensities(startIntensities);
            graphBefore.Repr(1.0d).WriteMRC($@"{outdir}\GraphBefore.mrc");

            Image ProjectorParticles = proj.ProjectToRealspace(particleRes, angles);
            Image ProjectorMasks = maskProj.ProjectToRealspace(particleRes, angles);
            ProjectorMasks.Binarize(0.9f);
            GPU.Normalize(ProjectorParticles.GetDevice(Intent.Read),
                            ProjectorParticles.GetDevice(Intent.Write),
                            (uint)ProjectorParticles.ElementsSliceReal,
                            (uint)angles.Count());

            ProjectorParticles.WriteMRC($@"{outdir}\ProjectorParticles.mrc");
            ProjectorMasks.WriteMRC($@"{outdir}\ProjectorMasks.mrc");



            int numIt = 20;
            float[][] itCorrs = Helper.ArrayOfFunction(i => new float[numPseudoparticles], numIt + 1);
            float[] err = new float[numIt];
            float[][] diff = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt);
            float[] newIntensities = new float[graph.Atoms.Count()];

            // The GT particles, as created by the pseudo projector
            Image PseudoProjectorParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

            IntPtr PseudoProjector = EntryPoint(inputVol.Dims, atomPositions, realIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());

            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(PseudoProjector, PseudoProjectorParticles.GetHost(Intent.Write)[k],null, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorParticles.WriteMRC($@"{outdir}\PseudoProjectorParticles.mrc");

            // The GT particles that the shifted graph creates

            Image PseudoProjectorMovedParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

            IntPtr MovedPseudoProjector = EntryPoint(inputVol.Dims, movedAtomPositions, realIntensities, graphMoved.Atoms[0].R, (uint)graphMoved.Atoms.Count());
            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(MovedPseudoProjector, PseudoProjectorMovedParticles.GetHost(Intent.Write)[k], null, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorMovedParticles.WriteMRC($@"{outdir}\PseudoProjectorMovedParticles.mrc");

            // The current projections that the reconstruction gives
            Image PseudoProjectorCurrProjections = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

            IntPtr PseudoReconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
            for (int l = 0; l < numPseudoparticles; l++)
            {
                GetProjection(PseudoReconstructor, PseudoProjectorCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
            }
            {
                float[] correlation = ImageProcessor.correlate(PseudoProjectorParticles, PseudoProjectorCurrProjections, ProjectorMasks);
                itCorrs[0] = correlation;
            }
            PseudoProjectorCurrProjections.WriteMRC($@"{outdir}\Projections_it0.mrc");


            // The actual reconstruction
            for (int k = 0; k < 0; k++)
            {
                err[k] = DoARTStepMoved(PseudoReconstructor, PseudoProjectorMovedParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), movedAtomPositions, 0.0f, 0.0f, (uint)numPseudoparticles);

                /* Get Current Projections */
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjection(PseudoReconstructor, PseudoProjectorCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }
                float[] correlation = ImageProcessor.correlate(PseudoProjectorParticles, PseudoProjectorCurrProjections, ProjectorMasks);
                itCorrs[k + 1] = correlation;
                PseudoProjectorCurrProjections.WriteMRC($@"{outdir}\Projections_it{k+1}.mrc");

                GetIntensities(PseudoReconstructor, newIntensities);
                for (int idx = 0; idx < newIntensities.Count(); idx++)
                {
                    diff[k][idx] = newIntensities[idx] - realIntensities[idx];
                }

            }

            using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itDiffs.txt"))
            {
                for (int k = 0; k < numIt; k++)
                {
                    file.WriteLine($"it: {k}");
                    for (int idx = 0; idx < newIntensities.Count(); idx++)
                    {
                        file.WriteLine($"{diff[k][idx]}");
                    }
                }
            }


            using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itErrs.txt"))
            {
                for (int k = 0; k < numIt; k++)
                {
                    file.WriteLine($"it: {k}");
                    file.WriteLine($"{err[k]}");
                }
            }


            using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itCorrs.txt"))
            {
                for (int l = 0; l < numParticles - 1; l++)
                {
                    file.Write($"{(itCorrs[0][l]).ToString(nfi)}, ");
                }
                file.Write($"{(itCorrs[0][numParticles - 1]).ToString(nfi)}\n");

                for (int k = 0; k < numIt; k++)
                {
                    file.WriteLine($"it: {k}");
                    for (int l = 0; l < numParticles-1; l++)
                    {
                        file.Write($"{(itCorrs[k + 1][l]).ToString(nfi)}, ");
                    }
                    file.Write($"{(itCorrs[k + 1][numParticles-1]).ToString(nfi)}\n");
                }
            }




            targetGraph.SetAtomIntensities(newIntensities);

            targetGraph.Repr(1.0d).WriteMRC($@"{outdir}\GraphAfter.mrc");

            
            /* Create CTFs*/





            {

                
                float[] outp = new float[PseudoProjectorMovedParticles.Dims.Elements()];
                float[] ctf = CTFs.GetHostContinuousCopy();
                convolve(PseudoProjectorMovedParticles.GetHostContinuousCopy(), ctf, outp, PseudoProjectorMovedParticles.Dims);
                Image convolved = new Image(outp, PseudoProjectorMovedParticles.Dims);
                convolved.WriteMRC($@"{outdir}\convolved.mrc");
                Image tmp = PseudoProjectorMovedParticles.AsFFT();
                tmp.Multiply(CTFs);
                //tmp.Divide(CTFs);
                Image PseudoProjectorMovedCTFParticles = tmp.AsIFFT();
                tmp.Dispose();

                PseudoProjectorMovedCTFParticles.WriteMRC($@"{outdir}\PseudoProjectorMovedCTFParticles.mrc");

                IntPtr MovedCTFPseudoProjector = EntryPoint(inputVol.Dims, movedAtomPositions, realIntensities, graphMoved.Atoms[0].R, (uint)graphMoved.Atoms.Count());


                // The current projections that the reconstruction gives
                Image PseudoProjectorCTFCurrProjections = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

                IntPtr CTFPseudoReconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjection(MovedCTFPseudoProjector, PseudoProjectorCTFCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }
                {
                    float[] correlation = ImageProcessor.correlate(PseudoProjectorMovedCTFParticles, PseudoProjectorCurrProjections, ProjectorMasks);
                    itCorrs[0] = correlation;
                }
                PseudoProjectorCurrProjections.WriteMRC($@"{outdir}\CTFprojections_it0.mrc");


                // The actual reconstruction
                for (int k = 0; k < numIt; k++)
                {
                    err[k] = DoARTStepMoved(PseudoReconstructor, PseudoProjectorMovedCTFParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), movedAtomPositions, 0.0f, 0.0f, (uint)numPseudoparticles);

                    /* Get Current Projections */
                    for (int l = 0; l < numPseudoparticles; l++)
                    {
                        GetProjection(PseudoReconstructor, PseudoProjectorCTFCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                    }
                    float[] correlation = ImageProcessor.correlate(PseudoProjectorParticles, PseudoProjectorCTFCurrProjections, ProjectorMasks);
                    itCorrs[k + 1] = correlation;
                    PseudoProjectorCurrProjections.WriteMRC($@"{outdir}\CTFprojections_it{k + 1}.mrc");

                    GetIntensities(PseudoReconstructor, newIntensities);
                    for (int idx = 0; idx < newIntensities.Count(); idx++)
                    {
                        diff[k][idx] = newIntensities[idx] - realIntensities[idx];
                    }

                }

            }

        }
    }
}
