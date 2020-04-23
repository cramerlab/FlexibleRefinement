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
        public static extern void GetProjectionCTF(IntPtr proj, float[] output, float[] output_nrm, float[] GaussTables, float[] GaussTables2, float border, float3 angles, float shiftX, float shiftY);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStep")]
        public static extern float DoARTStep(IntPtr proj, float[] Iexp, float3[] angles, float shiftX, float shiftY, uint numImages);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStepMoved")]
        public static extern float DoARTStepMoved(IntPtr proj, float[] Iexp, float3[] angles, float[] atomPositions, float shiftX, float shiftY, uint numImages);
        

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStepMovedCTF")]
        public static extern float DoARTStepMovedCTF(IntPtr proj, float[] Iexp, float3[] angles, float[] atomPositions, float[] GaussTables, float[] GaussTables2, float border, float shiftX, float shiftY, uint numImages);
        
        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStepMovedCTF_DB")]
        public static extern float DoARTStepMovedCTF_DB(IntPtr proj, float[] Iexp, float[] Itheo, float[] Icorr, float[] Idiff, float3[] angles, float[] atomPositions, float[] GaussTables, float[] GaussTables2, float border, float shiftX, float shiftY, uint numImages);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetIntensities")]
        public static extern void GetIntensities(IntPtr proj, float[] outp);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "convolve")]
        public static extern void convolve(float[] img, float[] ctf, float[] outp, int3 dims);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "getGaussianTableFull")]
        public static extern void getGaussianTableFull(float[] table, float sigma, int elements);


        public static Image FFT_slices(Image im, int plan=0)
        {
            Image[] ims = Helper.ArrayOfFunction(z => new Image(im.GetHost(Intent.Read)[z], new int3(im.Dims.X, im.Dims.Y, 1)), im.Dims.Z);
            Image[] FFTs = Helper.ArrayOfFunction(i => new Image(new int3(im.Dims.X, im.Dims.Y, 1), true, true, false), im.Dims.Z);
            for (int slice = 0; slice < im.Dims.Z; slice++)
            {
                GPU.FFT(ims[slice].GetDevice(Intent.Read), FFTs[slice].GetDevice(Intent.Write), im.Dims.Slice(), 1, plan);
                ims[slice].Dispose();
                FFTs[slice].FreeDevice();
            }
            Image FFT = Image.Stack(FFTs);
            FFT.FreeDevice();
            return FFT;
        }

        public static Image IFFT_slices(Image im, int plan = 0, bool normalize = false)
        {
            if (im.IsHalf || !im.IsComplex || !im.IsFT)
                throw new Exception("Data format not supported.");
            Image[] ims = Helper.ArrayOfFunction(z => new Image(im.GetHost(Intent.Read)[z], new int3(im.Dims.X, im.Dims.Y, 1), true, true, false), im.Dims.Z);
            Image[] IFFTs = Helper.ArrayOfFunction(i => new Image(new int3(im.Dims.X, im.Dims.Y, 1), false, false, false), im.Dims.Z);
            for (int slice = 0; slice < im.Dims.Z; slice++)
            {
                GPU.IFFT(ims[slice].GetDevice(Intent.Read), IFFTs[slice].GetDevice(Intent.Write), im.Dims.Slice(), 1, plan, normalize);
                ims[slice].Dispose();
                IFFTs[slice].FreeDevice();
            }
            Image IFFT = Image.Stack(IFFTs);
            IFFT.FreeDevice();
            return IFFT;
        }



        public static Image GetCTFCoords(int2 size, int2 originalSize, float pixelSize = 1, float pixelSizeDelta = 0, float pixelSizeAngle = 0)
        {
            Image rs = new Image(new int3(size.X, size.Y, 1),true);
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
                        rs.GetHost(Intent.Write)[0][y * (size.X / 2 + 1) + x] = r;
                    }

                CTFCoords = new Image(CTFCoordsData, new int3(size.X, size.Y, 1), true);
            }

            return CTFCoords;
        }


        static void Main(string[] args)
        {

            Image yyyy = Image.FromFile(@"D:\EMPIAR\10168\emd_4180.mrc");
            yyyy.Binarize(0.0004f);
            yyyy = yyyy.AsConvolvedGaussian(2.0f);
            yyyy.Binarize((float)(1.0f/Math.E));
            yyyy.WriteMRC(@"D:\EMPIAR\10168\emd_4180.mask.mrc");

            NumberFormatInfo nfi = new NumberFormatInfo();
            nfi.NumberDecimalSeparator = ".";
            Random rand = new Random();

            String inputName = "lowHighMix_small";

            String outdir = $@"D:\Software\FlexibleRefinement\bin\Debug\Refinement\current\{inputName}";

            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }
            int2 particleRes = new int2(40);
            String indir = $@"D:\Software\FlexibleRefinement\bin\Debug\Toy_Modulated\{inputName}\current_{particleRes.X}_{particleRes.Y}";

            Star TableIn = new Star(@"D:\florian_debug\particles.star");
            CTF[] CTFParams = TableIn.GetRelionCTF();

            Image inputVol = Image.FromFile($@"{indir}\startIm_fromGraph.mrc");
            Image inputMask = Image.FromFile($@"{indir}\startMask.mrc");

            AtomGraph graph = new AtomGraph($@"{indir}\StartGraph.graph", inputVol);
            AtomGraph graphMoved = new AtomGraph($@"{indir}\TargetGraph100.graph", inputVol);
            Projector proj = new Projector(inputVol, 2);
            Projector maskProj = new Projector(inputMask, 2);

            int numParticles = 500;
            int numAngles = numParticles;
            int numAnglesX = (int)Math.Ceiling(Math.Pow(numAngles, 1.0d / 3.0d));
            int numAnglesY = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            int numAnglesZ = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            numAngles = numAnglesX * numAnglesY * numAnglesZ;
            numParticles = numAngles;

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

            CTFParams = Helper.ArrayOfFunction(i => CTFParams[i % CTFParams.Length], numParticles);
            
            for (int i = 0; i < CTFParams.Length; i++)
                CTFParams[i].PixelSize = (decimal)3.0;// CTFParams[i].PixelSize* (decimal)2.2;


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
            int factor = 30;
            float sigma = graph.Atoms[0].R;

            //int2 gaussTableLayout = new int2(2 * factor * (int)Math.Ceiling(Math.Sqrt(2) * 4 * sigma), 1);



            Image noise = new Image(Simulation.simplexPerlinNoise(1, new int3(particleRes.X, particleRes.Y, numParticles)), new int3(particleRes.X, particleRes.Y, numParticles));

            int numPseudoparticles = numParticles;
            float[] projectionAngles = new float[numPseudoparticles * 3];
            for (int k = 0; k < numPseudoparticles; k++)
            {
                projectionAngles[3 * k] = angles[k].X;
                projectionAngles[3 * k + 1] = angles[k].Y;
                projectionAngles[3 * k + 2] = angles[k].Z;
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
            ProjectorParticles.FreeDevice();
            ProjectorMasks.FreeDevice();

            ProjectorMasks.Binarize(0.9f);
            GPU.Normalize(ProjectorParticles.GetDevice(Intent.Read),
                            ProjectorParticles.GetDevice(Intent.Write),
                            (uint)ProjectorParticles.ElementsSliceReal,
                            (uint)angles.Count());
            ProjectorParticles.FreeDevice();
            ProjectorParticles.WriteMRC($@"{outdir}\ProjectorParticles.mrc");
            ProjectorMasks.WriteMRC($@"{outdir}\ProjectorMasks.mrc");

            int numIt = 20;

            float[] newIntensities = new float[graph.Atoms.Count()];

            IntPtr PseudoProjectorGT = EntryPoint(inputVol.Dims, atomPositions, realIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
            IntPtr PseudoProjectorGTMoved = EntryPoint(inputVol.Dims, movedAtomPositions, realIntensities, graphMoved.Atoms[0].R, (uint)graphMoved.Atoms.Count());

            // The GT particles, as created by the pseudo projector
            Image PseudoProjectorGTParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(PseudoProjectorGT, PseudoProjectorGTParticles.GetHost(Intent.Write)[k], null, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorGTParticles.Add(noise);
            PseudoProjectorGTParticles.WriteMRC($@"{outdir}\PseudoProjectorGTParticles.mrc");

            // The GT particles for the moved reconstruction
            Image PseudoProjectorGTMovedParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(PseudoProjectorGTMoved, PseudoProjectorGTMovedParticles.GetHost(Intent.Write)[k], null, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorGTMovedParticles.Add(noise);
            PseudoProjectorGTMovedParticles.WriteMRC($@"{outdir}\PseudoProjectorGTMovedParticles.mrc");


            /*Unmoved Reconstruction */
            if(!File.Exists($@"{outdir}\GraphAfter.xyz"))
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

                CurrProj.WriteMRC($@"{outdir}\PseudoProjectorCurrProjections_it0.mrc");


                // The actual reconstruction
                Stopwatch stopWatch = new Stopwatch();
                stopWatch.Start();
                for (int k = 0; k < numIt; k++)
                {
                    itErrs[k] = DoARTStep(Reconstructor, PseudoProjectorGTParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), 0.0f, 0.0f, (uint)numPseudoparticles);

                    /* Get Current Projections */
                    for (int l = 0; l < numPseudoparticles; l++)
                    {
                        GetProjection(Reconstructor, CurrProj.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                    }

                    itCorrs[k + 1] = ImageProcessor.correlate(PseudoProjectorGTParticles, CurrProj, ProjectorMasks);
                    CurrProj.WriteMRC($@"{outdir}\PseudoProjectorCurrProjections_it{k + 1}.mrc");

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
                    file.WriteLine($"it: {0}");
                    for (int l = 0; l < numParticles - 1; l++)
                    {
                        file.Write($"{(itCorrs[0][l]).ToString(nfi)} ");
                    }
                    file.Write($"{(itCorrs[0][numParticles - 1]).ToString(nfi)}\n");

                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k+1}");
                        for (int l = 0; l < numParticles - 1; l++)
                        {
                            file.Write($"{(itCorrs[k + 1][l]).ToString(nfi)} ");
                        }
                        file.Write($"{(itCorrs[k + 1][numParticles - 1]).ToString(nfi)}\n");
                    }
                }




                targetGraph.SetAtomIntensities(newIntensities);

                targetGraph.Repr(1.0d).WriteMRC($@"{outdir}\GraphAfter.mrc");
                targetGraph.save($@"{outdir}\GraphAfter.graph");
                targetGraph.save($@"{outdir}\GraphAfter.xyz");
            }

            /*Moved Reconstruction */
            if (!File.Exists($@"{outdir}\GraphAfterMovedReconstruction.xyz"))
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
                for (int k = 0; k < numIt; k++)
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
                    file.WriteLine($"it: {0}");
                    for (int l = 0; l < numParticles - 1; l++)
                    {
                        file.Write($"{(itCorrsMoved[0][l]).ToString(nfi)} ");
                    }
                    file.Write($"{(itCorrsMoved[0][numParticles - 1]).ToString(nfi)}\n");

                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k+1}");
                        for (int l = 0; l < numParticles - 1; l++)
                        {
                            file.Write($"{(itCorrsMoved[k + 1][l]).ToString(nfi)} ");
                        }
                        file.Write($"{(itCorrsMoved[k + 1][numParticles - 1]).ToString(nfi)}\n");
                    }
                }




                targetGraph.SetAtomIntensities(newIntensities);

                targetGraph.Repr(1.0d).WriteMRC($@"{outdir}\GraphAfterMovedReconstruction.mrc");
                targetGraph.save($@"{outdir}\GraphAfterMovedReconstruction.graph");
                targetGraph.save($@"{outdir}\GraphAfterMovedReconstruction.xyz");
            }

            int2 gaussTableLayout = new int2(factor * particleRes.X, factor * particleRes.Y);
            float border = (float) (Math.Min(3*sigma, ProjectorParticles.Dims.X / 2 / Math.Sqrt(2)));
            Image CTFCoordsGauss = GetCTFCoords(new int2(gaussTableLayout.X, gaussTableLayout.Y), new int2(particleRes.X, particleRes.Y), 1.0f);
            Image CTFsGauss = new Image(new int3(gaussTableLayout.X, gaussTableLayout.Y, numParticles), true);
            GPU.CreateCTF(CTFsGauss.GetDevice(Intent.Write),
                            CTFCoordsGauss.GetDevice(Intent.Read),
                            (uint)CTFCoordsGauss.ElementsSliceComplex,
                            CTFParams.Select(p => p.ToStruct()).ToArray(),
                            false,
                            (uint)numParticles);
            CTFCoordsGauss.Dispose();
            CTFsGauss.WriteMRC($@"{outdir}\CTFsGauss.mrc");
            Image Gauss = new Image(new int3(gaussTableLayout.X, gaussTableLayout.Y, numParticles));
            Gauss.TransformValues((x, y, z, v) => (float)Math.Exp(-0.5 * (Math.Pow((x - gaussTableLayout.X / 2.0) / (double)factor, 2) + Math.Pow((y - gaussTableLayout.Y / 2.0) / (double)factor,2) )/Math.Pow(sigma, 2.0)));
            Gauss.WriteMRC($@"{outdir}\Gauss.mrc");
            Image GaussFT = FFT_slices(Gauss);
            Image GaussIFT = IFFT_slices(GaussFT, 0, true);
            GaussFT.Add(CTFsGauss);
            Image GaussConvolved = IFFT_slices(GaussFT, 0, true);
            GaussConvolved.WriteMRC($@"{outdir}\GaussConvolved.mrc");

            float[][] gaussConvolvedFlat = GaussConvolved.GetHost(Intent.Read);
            float[] gaussConvolved = new float[numParticles * factor * particleRes.X / 2];
            float[] gaussConvolved2 = new float[numParticles * factor * particleRes.X / 2];
            float[][] gaussConvolved_2D = Helper.ArrayOfFunction(i => new float[factor * particleRes.X / 2], numParticles);
            float[][] gaussConvolved2_2D = Helper.ArrayOfFunction(i => new float[factor * particleRes.X / 2], numParticles);
            for (int zz = 0; zz < numParticles; zz++)
            {
                for (int j = 0; j < gaussTableLayout.X / 2; j++)
                {
                    gaussConvolved[zz * gaussTableLayout.X / 2 + j] = gaussConvolvedFlat[zz][gaussTableLayout.Y / 2 * gaussTableLayout.X + gaussTableLayout.X / 2 + j];
                    gaussConvolved_2D[zz][j] = gaussConvolvedFlat[zz][gaussTableLayout.Y/2*gaussTableLayout.X + gaussTableLayout.X / 2 + j];
                    gaussConvolved2[zz * gaussTableLayout.X / 2 + j] = gaussConvolvedFlat[zz][gaussTableLayout.Y / 2 * gaussTableLayout.X + gaussTableLayout.X / 2 + j] * gaussConvolvedFlat[zz][gaussTableLayout.Y / 2 * gaussTableLayout.X + gaussTableLayout.X / 2 + j];
                    gaussConvolved2_2D[zz][j] = gaussConvolvedFlat[zz][gaussTableLayout.Y / 2 * gaussTableLayout.X + gaussTableLayout.X / 2 + j] * gaussConvolvedFlat[zz][gaussTableLayout.Y / 2 * gaussTableLayout.X + gaussTableLayout.X / 2 + j];
                }
            }

            Image PseudoProjectorGTCTFParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjectionCTF(PseudoProjectorGT, PseudoProjectorGTCTFParticles.GetHost(Intent.Write)[k], null, gaussConvolved_2D[k], gaussConvolved2_2D[k],border, angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }
            PseudoProjectorGTCTFParticles.Add(noise);
            PseudoProjectorGTCTFParticles.WriteMRC($@"{outdir}\PseudoProjectorGTCTFParticles.mrc");

            Image GTCTFParticles = new Image(PseudoProjectorGTParticles.Dims);
            {
                Image ftTmp = PseudoProjectorGTParticles.AsFFT();
                ftTmp.Multiply(CTFs);
                GTCTFParticles = ftTmp.AsIFFT(false, 0, true);
            }
            GTCTFParticles.Add(noise);
            GTCTFParticles.WriteMRC($@"{outdir}\GTCTFParticles.mrc");
            Image testImage2 = new Image(new int3(particleRes.X, particleRes.Y, numParticles));
            testImage2.TransformValues((x, y, z, v) =>
            {
                int2 center = new int2(testImage2.Dims.X / 2, testImage2.Dims.Y / 2);
                int2 center2 = new int2(3 * testImage2.Dims.X / 4, testImage2.Dims.Y / 2);
                double res = 0.0;
                double r1 = Math.Sqrt(Math.Pow(x - center.X, 2) + Math.Pow(y - center.Y, 2));
                int xx1 = (int)(((x - center.X)) * factor + gaussTableLayout.X / 2.0);
                int yy1 = (int)(((y - center.Y)) * factor + gaussTableLayout.Y / 2.0);

                if (xx1 < Gauss.Dims.X && xx1 > 0 && yy1 < Gauss.Dims.Y && yy1 > 0)
                {
                    res += Gauss.GetHost(Intent.Read)[z][yy1*Gauss.Dims.X+xx1];
                }
                /*
                double r2 = Math.Sqrt(Math.Pow(x - center2.X, 2) + Math.Pow(y - center2.Y, 2));
                int idx2 = (int)Math.Round(r2 * factor);
                if (idx2 < Gauss.Dims.X / 2)
                {
                    res += Gauss.GetHost(Intent.Read)[z][gaussTableLayout.X / 2 + idx2];
                }*/
                return (float)res;

            });
            Image testImage2SelfConvolved = new Image(new int3(particleRes.X, particleRes.Y, numParticles));
            testImage2SelfConvolved.TransformValues((x, y, z, v) =>
            {
                int2 center = new int2(testImage2.Dims.X/2, testImage2.Dims.Y/2);
                int2 center2 = new int2(3*testImage2.Dims.X / 4, testImage2.Dims.Y / 2);
                double res = 0.0;
                int xx1 = (int)(((x - center.X) ) * factor + gaussTableLayout.X / 2.0);
                int yy1 = (int)(((y - center.Y) ) * factor + gaussTableLayout.Y / 2.0);

                if (xx1 < GaussConvolved.Dims.X && xx1 > 0 && yy1 < GaussConvolved.Dims.Y && yy1 > 0)
                {
                    res += GaussConvolved.GetHost(Intent.Read)[z][yy1 * Gauss.Dims.X + xx1];
                }

                /*double r2 = Math.Sqrt(Math.Pow(x - center2.X, 2) + Math.Pow(y - center2.Y, 2));
                int idx2 = (int)Math.Round(r2 * factor);
                if (idx2 < GaussConvolved.Dims.X / 2)
                {
                    res += GaussConvolved.GetHost(Intent.Read)[z][gaussTableLayout.X / 2 + idx2];
                }*/
                return (float)res;

            });

            Image testImage2FT = testImage2.AsFFT();
            testImage2FT.Multiply(CTFs);

            Image testImage2Convolved = testImage2FT.AsIFFT(false, 0, true);
            //testImage2.Normalize();
            //testImage2Convolved.Normalize();
            //testImage2SelfConvolved.Normalize();
            testImage2Convolved.WriteMRC($@"{outdir}\testImage2Convolved.mrc");
            testImage2.WriteMRC($@"{outdir}\TestImage2.mrc");
            testImage2SelfConvolved.WriteMRC($@"{outdir}\testImage2SelfConvolved.mrc");
            /* CTF Reconstruction */
            if (!File.Exists($@"{outdir}\GraphAfterCTFReconstruction.xyz"))
            {
                float[][] itCorrsCTF = Helper.ArrayOfFunction(i => new float[numPseudoparticles], numIt + 1);
                float[] itErrsCTF = new float[numIt];
                float[][] itDiffsCTF = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt);
                float[][] itintsCTF = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt + 1);
                float[] outp = new float[PseudoProjectorGTParticles.Dims.Elements()];
                float[] ctf = CTFs.GetHostContinuousCopy();

                IntPtr CTFPseudoreconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());

                // The current projections that the reconstruction gives
                Image PseudoProjectorCTFCurrProjections = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

                IntPtr CTFPseudoReconstructor = EntryPoint(inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjectionCTF(CTFPseudoreconstructor, PseudoProjectorCTFCurrProjections.GetHost(Intent.Write)[l], null, gaussConvolved, gaussConvolved2, border, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }
                {
                    float[] correlation = ImageProcessor.correlate(GTCTFParticles, PseudoProjectorCTFCurrProjections, ProjectorMasks);
                    itCorrsCTF[0] = correlation;
                }
                PseudoProjectorCTFCurrProjections.WriteMRC($@"{outdir}\CTFprojections_it0.mrc");

                Stopwatch stopWatch = new Stopwatch();
                stopWatch.Start();

                float[] ItheoData = new float[GTCTFParticles.Dims.X * GTCTFParticles.Dims.Y * GTCTFParticles.Dims.Z];
                float[] IcorrData = new float[GTCTFParticles.Dims.X * GTCTFParticles.Dims.Y * GTCTFParticles.Dims.Z];
                float[] IdiffData = new float[GTCTFParticles.Dims.X * GTCTFParticles.Dims.Y * GTCTFParticles.Dims.Z];

                GetIntensities(CTFPseudoreconstructor, newIntensities);
                float tmp = newIntensities[0];
                for (int idx = 0; idx < newIntensities.Count(); idx++)
                {
                    itintsCTF[0][idx] = newIntensities[idx];
                }
                // The actual reconstruction
                for (int k = 0; k < numIt; k++)
                {

                    itErrsCTF[k] = DoARTStepMovedCTF_DB(CTFPseudoreconstructor, GTCTFParticles.GetHostContinuousCopy(), ItheoData, IcorrData, IdiffData, Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), atomPositions, gaussConvolved, gaussConvolved2, border, 0.0f, 0.0f, (uint)numPseudoparticles);
                    Image currentItheo = new Image(ItheoData, GTCTFParticles.Dims);
                    Image currentIcorr = new Image(ItheoData, GTCTFParticles.Dims);
                    Image currentIdiff = new Image(ItheoData, GTCTFParticles.Dims);
                    currentItheo.WriteMRC($@"{outdir}\currentItheo_it{k + 1}.mrc");
                    currentIcorr.WriteMRC($@"{outdir}\currentIcorr_it{k + 1}.mrc");
                    currentIdiff.WriteMRC($@"{outdir}\currentIdiff_it{k + 1}.mrc");
                    /* Get Current Projections */
                    for (int l = 0; l < numPseudoparticles; l++)
                    {
                        //GetProjection(MovedCTFPseudoProjector, PseudoProjectorMovedCTFCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                        GetProjectionCTF(CTFPseudoreconstructor, PseudoProjectorCTFCurrProjections.GetHost(Intent.Write)[l], null, gaussConvolved_2D[l], gaussConvolved2_2D[l], border, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                    }
                    float[] correlation = ImageProcessor.correlate(GTCTFParticles, PseudoProjectorCTFCurrProjections, ProjectorMasks);
                    itCorrsCTF[k + 1] = correlation;
                    PseudoProjectorCTFCurrProjections.WriteMRC($@"{outdir}\CTFprojections_it{k + 1}.mrc");

                    GetIntensities(CTFPseudoreconstructor, newIntensities);

                    for (int idx = 0; idx < newIntensities.Count(); idx++)
                    {
                        itDiffsCTF[k][idx] = newIntensities[idx] - realIntensities[idx];
                        itintsCTF[k + 1][idx] = newIntensities[idx];
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

                using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}\itIntsCTF.txt"))
                {
                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k}");
                        for (int idx = 0; idx < newIntensities.Count(); idx++)
                        {
                            file.WriteLine($"{itintsCTF[k][idx]}");
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
                    file.WriteLine($"it: {0}");
                    for (int l = 0; l < numParticles - 1; l++)
                    {
                        file.Write($"{(itCorrsCTF[0][l]).ToString(nfi)} ");
                    }
                    file.Write($"{(itCorrsCTF[0][numParticles - 1]).ToString(nfi)}\n");

                    for (int k = 0; k < numIt; k++)
                    {
                        file.WriteLine($"it: {k+1}");
                        for (int l = 0; l < numParticles - 1; l++)
                        {
                            file.Write($"{(itCorrsCTF[k + 1][l]).ToString(nfi)} ");
                        }
                        file.Write($"{(itCorrsCTF[k + 1][numParticles - 1]).ToString(nfi)}\n");
                    }
                }

                targetGraph.SetAtomIntensities(newIntensities);

                targetGraph.Repr(1.0d).WriteMRC($@"{outdir}\GraphAfterCTFReconstruction.mrc");
                targetGraph.save($@"{outdir}\GraphAfterCTFReconstruction.graph");
                targetGraph.save($@"{outdir}\GraphAfterCTFReconstruction.xyz");



            }

        }
    }
}
