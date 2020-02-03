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
using System.Diagnostics;

namespace FlexibleRefinement
{
    [SuppressUnmanagedCodeSecurity]
    class Program
    {

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EntryPoint")]
        public static extern IntPtr EntryPoint(int3 dims, float[] atomCenters, float[] atomWeights, float rAtoms, uint nAtoms);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetProjection")]
        public static extern void GetProjection(IntPtr proj, float[] output, float[] output_nrm, float3 angles, float shiftX, float shiftY);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetProjectionCTF")]
        public static extern void GetProjectionCTF(IntPtr proj, float[] output, float[] output_nrm, float[] GaussTables, float[] GaussTables2, float3 angles, float shiftX, float shiftY);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStep")]
        public static extern float DoARTStep(IntPtr proj, float[] Iexp, float3[] angles, float shiftX, float shiftY, uint numImages);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStepMoved")]
        public static extern float DoARTStepMoved(IntPtr proj, float[] Iexp, float3[] angles, float[] atomPositions, float shiftX, float shiftY, uint numImages);
        

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStepMovedCTF")]
        public static extern float DoARTStepMovedCTF(IntPtr proj, float[] Iexp, float3[] angles, float[] atomPositions, float[] GaussTables, float[] GaussTables2, float shiftX, float shiftY, uint numImages);

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


            int numParticles = 1000;
            int numAngles = numParticles;
            int numAnglesX = (int)Math.Ceiling(Math.Pow(numAngles, 1.0d / 3.0d));
            int numAnglesY = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            int numAnglesZ = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            numAngles = numAnglesX * numAnglesY * numAnglesZ;
            numParticles = numAngles;

            CTFParams = Helper.ArrayOfFunction(i => CTFParams[i % CTFParams.Length], numParticles);
            int2 particleRes = new int2(100);
            for (int i = 0; i < CTFParams.Length; i++)
                CTFParams[i].PixelSize = (decimal)1.8;// CTFParams[i].PixelSize* (decimal)2.2;


            for (int idx = 0; idx < CTFParams.Length; idx++)
            {
                CTFParams[idx].Defocus = (decimal)((0.4 * rand.NextDouble() - 0.2) + 1);
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
            int factor = 1000;
            float sigma = 0.8f;

            //int2 gaussTableLayout = new int2(2 * factor * (int)Math.Ceiling(Math.Sqrt(2) * 4 * sigma), 1);
            




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

            float[] newIntensities = new float[graph.Atoms.Count()];

            

            IntPtr PseudoProjector = EntryPoint(inputVol.Dims, atomPositions, realIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
            IntPtr MovedPseudoProjector = EntryPoint(inputVol.Dims, movedAtomPositions, realIntensities, graphMoved.Atoms[0].R, (uint)graphMoved.Atoms.Count());

            // The GT particles, as created by the pseudo projector
            Image PseudoProjectorGTParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(PseudoProjector, PseudoProjectorGTParticles.GetHost(Intent.Write)[k], null, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorGTParticles.WriteMRC($@"{outdir}\PseudoProjectorGTParticles.mrc");

            // The GT particles for the moved reconstruction
            Image PseudoProjectorGTMovedParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(MovedPseudoProjector, PseudoProjectorGTMovedParticles.GetHost(Intent.Write)[k], null, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorGTMovedParticles.WriteMRC($@"{outdir}\PseudoProjectorGTMovedParticles.mrc");


            /*Unmoved Reconstruction */
            {
                float[][] itCorrs = Helper.ArrayOfFunction(i => new float[numPseudoparticles], numIt + 1);
                float[] itErrs = new float[numIt];
                float[][] itDiffs = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt);


                // The current projections that the reconstruction gives
                Image CurrProj = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

                IntPtr Reconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
                //Unmoved projection
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjection(Reconstructor, CurrProj.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }

                itCorrs[0] = ImageProcessor.correlate(PseudoProjectorGTParticles, CurrProj, ProjectorMasks); ;

                CurrProj.WriteMRC($@"{outdir}\PseudoProjectorCurrMovedProjections_it0.mrc");


                // The actual reconstruction
                Stopwatch stopWatch = new Stopwatch();
                stopWatch.Start();
                for (int k = 0; k < numIt; k++)
                {
                    itErrs[k] = DoARTStepMoved(Reconstructor, PseudoProjectorGTMovedParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), movedAtomPositions, 0.0f, 0.0f, (uint)numPseudoparticles);

                    /* Get Current Projections */
                    for (int l = 0; l < numPseudoparticles; l++)
                    {
                        GetProjection(Reconstructor, CurrProj.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                    }

                    itCorrs[k + 1] = ImageProcessor.correlate(PseudoProjectorGTParticles, CurrProj, ProjectorMasks);
                    CurrProj.WriteMRC($@"{outdir}\PseudoProjectorCurrMovedProjections_it{k + 1}.mrc");

                    GetIntensities(Reconstructor, newIntensities);
                    for (int idx = 0; idx < newIntensities.Count(); idx++)
                    {
                        itDiffs[k][idx] = newIntensities[idx] - realIntensities[idx];
                    }

                }
                stopWatch.Stop();
                TimeSpan ts = stopWatch.Elapsed;
                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}", ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds / 10);
                Console.WriteLine("PseudoProjectorMovedParticles Reconstruction took " + elapsedTime);

                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itDiffs.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        for (int idx = 0; idx < newIntensities.Count(); idx++)
                        {
                            file.WriteLine($"{itDiffs[k][idx]}");
                        }
                    }
                }


                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itErrs.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        file.WriteLine($"{itErrs[k]}");
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
                        for (int l = 0; l < numParticles - 1; l++)
                        {
                            file.Write($"{(itCorrs[k + 1][l]).ToString(nfi)}, ");
                        }
                        file.Write($"{(itCorrs[k + 1][numParticles - 1]).ToString(nfi)}\n");
                    }
                }




                targetGraph.SetAtomIntensities(newIntensities);

                targetGraph.Repr(1.0d).WriteMRC($@"{outdir}\GraphAfter.mrc");
            }

            /*Moved Reconstruction */
            {
                float[][] itCorrsMoved = Helper.ArrayOfFunction(i => new float[numPseudoparticles], numIt + 1);
                float[] itErrsMoved = new float[numIt];
                float[][] itDiffsMoved = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt);
                 

                // The current projections that the reconstruction gives
                Image CurrProj = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

                IntPtr Reconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
                //Unmoved projection
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjection(Reconstructor, CurrProj.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }
                
                itCorrsMoved[0] = ImageProcessor.correlate(PseudoProjectorGTParticles, CurrProj, ProjectorMasks); ;
                
                CurrProj.WriteMRC($@"{outdir}\PseudoProjectorCurrMovedProjections_it0.mrc");


                // The actual reconstruction
                Stopwatch stopWatch = new Stopwatch();
                stopWatch.Start();
                for (int k = 0; k < 0; k++)
                {
                    itErrsMoved[k] = DoARTStepMoved(Reconstructor, PseudoProjectorGTMovedParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), movedAtomPositions, 0.0f, 0.0f, (uint)numPseudoparticles);

                    /* Get Current Projections */
                    for (int l = 0; l < numPseudoparticles; l++)
                    {
                        GetProjection(Reconstructor, CurrProj.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                    }

                    itCorrsMoved[k + 1] = ImageProcessor.correlate(PseudoProjectorGTParticles, CurrProj, ProjectorMasks);
                    CurrProj.WriteMRC($@"{outdir}\PseudoProjectorCurrMovedProjections_it{k + 1}.mrc");

                    GetIntensities(Reconstructor, newIntensities);
                    for (int idx = 0; idx < newIntensities.Count(); idx++)
                    {
                        itDiffsMoved[k][idx] = newIntensities[idx] - realIntensities[idx];
                    }

                }
                stopWatch.Stop();
                TimeSpan ts = stopWatch.Elapsed;
                string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}", ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds / 10);
                Console.WriteLine("PseudoProjectorMovedParticles Reconstruction took " + elapsedTime);

                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itDiffsMoved.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        for (int idx = 0; idx < newIntensities.Count(); idx++)
                        {
                            file.WriteLine($"{itDiffsMoved[k][idx]}");
                        }
                    }
                }


                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itErrsMoved.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        file.WriteLine($"{itErrsMoved[k]}");
                    }
                }


                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itCorrsMoved.txt"))
                {
                    for (int l = 0; l < numParticles - 1; l++)
                    {
                        file.Write($"{(itCorrsMoved[0][l]).ToString(nfi)}, ");
                    }
                    file.Write($"{(itCorrsMoved[0][numParticles - 1]).ToString(nfi)}\n");

                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        for (int l = 0; l < numParticles - 1; l++)
                        {
                            file.Write($"{(itCorrsMoved[k + 1][l]).ToString(nfi)}, ");
                        }
                        file.Write($"{(itCorrsMoved[k + 1][numParticles - 1]).ToString(nfi)}\n");
                    }
                }




                targetGraph.SetAtomIntensities(newIntensities);

                targetGraph.Repr(1.0d).WriteMRC($@"{outdir}\GraphAfterMovedReconstruction.mrc");
            }
            
            /* CTF Reconstruction */
            {
                float[][] itCorrsCTF = Helper.ArrayOfFunction(i => new float[numPseudoparticles], numIt + 1);
                float[] itErrsCTF = new float[numIt];
                float[][] itDiffsCTF = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt);

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
                Gauss.TransformValues((x, y, z, v) => (float)(1 / (2 * Math.PI * sigma * sigma) * Math.Exp(-0.5 * Math.Pow(((x - gaussTableLayout.X / 2) / factor) / sigma, 2))));

                Image GaussFT = Gauss.AsFFT();
                Image GaussIFT = GaussFT.AsIFFT(false, 0, true);
                GaussFT.Multiply(CTFsGauss);
                Image GaussConvolved = GaussFT.AsIFFT(false, 0, true);

                float[] gaussConvolvedFlat = GaussConvolved.GetHostContinuousCopy();
                float[] gaussConvolved = new float[numParticles * gaussTableLayout.X / 2];
                float[] gaussConvolved2 = new float[numParticles * gaussTableLayout.X / 2];
                float[][] gaussConvolved_2D = Helper.ArrayOfFunction(i => new float[gaussTableLayout.X / 2], numParticles);
                float[][] gaussConvolved2_2D = Helper.ArrayOfFunction(i => new float[gaussTableLayout.X / 2], numParticles);
                for (int zz = 0; zz < numParticles; zz++)
                {
                    for (int j = 0; j < gaussTableLayout.X / 2; j++)
                    {
                        gaussConvolved[zz * gaussTableLayout.X / 2 + j] = gaussConvolvedFlat[gaussTableLayout.X * zz + gaussTableLayout.X / 2 + j];
                        gaussConvolved_2D[zz][gaussTableLayout.X / 2 + j] = gaussConvolvedFlat[gaussTableLayout.X * zz + gaussTableLayout.X / 2 + j];
                        gaussConvolved2[zz * gaussTableLayout.X / 2 + j] = gaussConvolvedFlat[gaussTableLayout.X * zz + gaussTableLayout.X / 2 + j] * gaussConvolvedFlat[gaussTableLayout.X * zz + gaussTableLayout.X / 2 + j];
                        gaussConvolved2_2D[zz][gaussTableLayout.X / 2 + j] = gaussConvolvedFlat[gaussTableLayout.X * zz + gaussTableLayout.X / 2 + j] * gaussConvolvedFlat[gaussTableLayout.X * zz + gaussTableLayout.X / 2 + j];
                    }
                }

                float[] outp = new float[PseudoProjectorGTParticles.Dims.Elements()];
                float[] ctf = CTFs.GetHostContinuousCopy();
                convolve(PseudoProjectorGTMovedParticles.GetHostContinuousCopy(), ctf, outp, PseudoProjectorGTMovedParticles.Dims);
                Image convolved = new Image(outp, PseudoProjectorGTMovedParticles.Dims);
                convolved.WriteMRC($@"{outdir}\convolved.mrc");

                Image PseudoProjectorGTMovedCTFParticles = new Image(PseudoProjectorGTMovedParticles.Dims);
                {
                    Image ftTmp = PseudoProjectorGTMovedParticles.AsFFT();
                    ftTmp.Multiply(CTFs);
                    PseudoProjectorGTMovedCTFParticles = ftTmp.AsIFFT(false, 0, true);
                }

                PseudoProjectorGTMovedCTFParticles.WriteMRC($@"{outdir}\PseudoProjectorMovedCTFParticles.mrc");

                IntPtr MovedCTFPseudoProjector = EntryPoint(inputVol.Dims, movedAtomPositions, realIntensities, graphMoved.Atoms[0].R, (uint)graphMoved.Atoms.Count());


                // The current projections that the reconstruction gives
                Image PseudoProjectorMovedCTFCurrProjections = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

                IntPtr CTFPseudoReconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjectionCTF(MovedCTFPseudoProjector, PseudoProjectorMovedCTFCurrProjections.GetHost(Intent.Write)[l], null, gaussConvolved_2D[l], gaussConvolved2_2D[l], angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }
                {
                    float[] correlation = ImageProcessor.correlate(PseudoProjectorGTMovedCTFParticles, PseudoProjectorMovedCTFCurrProjections, ProjectorMasks);
                    itCorrsCTF[0] = correlation;
                }
                PseudoProjectorMovedCTFCurrProjections.WriteMRC($@"{outdir}\CTFprojections_it0.mrc");

                Stopwatch stopWatch = new Stopwatch();
                stopWatch.Start();
                // The actual reconstruction
                for (int k = 0; k < numIt; k++)
                {
                    itErrsCTF[k] = DoARTStepMovedCTF(MovedCTFPseudoProjector, PseudoProjectorGTMovedCTFParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), movedAtomPositions, gaussConvolved, gaussConvolved2, 0.0f, 0.0f, (uint)numPseudoparticles);

                    /* Get Current Projections */
                    for (int l = 0; l < numPseudoparticles; l++)
                    {
                        GetProjection(MovedCTFPseudoProjector, PseudoProjectorMovedCTFCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                    }
                    float[] correlation = ImageProcessor.correlate(PseudoProjectorGTParticles, PseudoProjectorMovedCTFCurrProjections, ProjectorMasks);
                    itCorrsCTF[k + 1] = correlation;
                    PseudoProjectorMovedCTFCurrProjections.WriteMRC($@"{outdir}\CTFprojections_it{k + 1}.mrc");

                    GetIntensities(MovedCTFPseudoProjector, newIntensities);
                    for (int idx = 0; idx < newIntensities.Count(); idx++)
                    {
                        itDiffsCTF[k][idx] = newIntensities[idx] - realIntensities[idx];
                    }

                }
                stopWatch.Stop();
                TimeSpan ts = stopWatch.Elapsed;
                String elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00}", ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds / 10);
                Console.WriteLine("PseudoProjectorMovedCTFParticles Reconstruction took " + elapsedTime);

                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itDiffsCTF.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        for (int idx = 0; idx < newIntensities.Count(); idx++)
                        {
                            file.WriteLine($"{itDiffsCTF[k][idx]}");
                        }
                    }
                }


                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itErrsCTF.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        file.WriteLine($"{itErrsCTF[k]}");
                    }
                }


                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itCorrsCTF.txt"))
                {
                    for (int l = 0; l < numParticles - 1; l++)
                    {
                        file.Write($"{(itCorrsCTF[0][l]).ToString(nfi)}, ");
                    }
                    file.Write($"{(itCorrsCTF[0][numParticles - 1]).ToString(nfi)}\n");

                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        for (int l = 0; l < numParticles - 1; l++)
                        {
                            file.Write($"{(itCorrsCTF[k + 1][l]).ToString(nfi)}, ");
                        }
                        file.Write($"{(itCorrsCTF[k + 1][numParticles - 1]).ToString(nfi)}\n");
                    }
                }



            }

        }
    }
}
