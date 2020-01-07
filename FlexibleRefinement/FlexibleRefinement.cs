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

namespace FlexibleRefinement
{
    [SuppressUnmanagedCodeSecurity]
    class Program
    {

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "EntryPoint")]
        public static extern IntPtr EntryPoint(float[] projections, float[] angles, int3 dims, float[] atomCenters, float[] atomWeights, float rAtoms, uint nAtoms);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetProjection")]
        public static extern void GetProjection(IntPtr proj, float[] output, float[] output_nrm, float3 angles, float shiftX, float shiftY);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "DoARTStep")]
        public static extern float DoARTStep(IntPtr proj, float[] Iexp, float3[] angles, float shiftX, float shiftY, uint numImages);

        [DllImport("PseudoRefinement.dll", CharSet = CharSet.Ansi, SetLastError = true, CallingConvention = CallingConvention.StdCall, EntryPoint = "GetIntensities")]
        public static extern void GetIntensities(IntPtr proj, float[] outp);


        static void Main(string[] args)
        {

            String outdir = @"D:\Software\FlexibleRefinement\bin\Debug\Refinement\";
            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }




            Star TableIn = new Star(@"D:\florian_debug\particles.star");
            CTF[] CTFParams = TableIn.GetRelionCTF();


            Image inputVol = Image.FromFile($@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy_Modulated\100\startIm.mrc");
            Image inputMask = Image.FromFile($@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy_Modulated\100\startMask.mrc");

            AtomGraph graph = new AtomGraph(@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy_Modulated\100\StartGraph.graph", inputVol);
            Projector proj = new Projector(inputVol, 2);
            Projector maskProj = new Projector(inputMask, 2);


            int numParticles = 300;
            int numAngles = numParticles;
            CTFParams.Take(numParticles).ToArray();
            int2 particleRes = new int2(100);
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




            Image ProjectorParticles = proj.ProjectToRealspace(particleRes, angles);
            Image ProjectorMasks = maskProj.ProjectToRealspace(particleRes, angles);
            ProjectorMasks.Binarize(0.9f);
            GPU.Normalize(ProjectorParticles.GetDevice(Intent.Read),
                            ProjectorParticles.GetDevice(Intent.Write),
                            (uint)ProjectorParticles.ElementsSliceReal,
                            (uint)angles.Count());

            ProjectorParticles.WriteMRC($@"{outdir}ProjectorParticles.mrc");
            ProjectorMasks.WriteMRC($@"{outdir}ProjectorMasks.mrc");

            int numPseudoparticles = numParticles;
            Image PseudoProjectorParticles = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            Image PseudoProjectorCurrProjections = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            Image PseudoProjectorCurrProjectionsNorm = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));
            Image PseudoProjectorParticlesNorm = new Image(new int3(ProjectorParticles.Dims.X, ProjectorParticles.Dims.Y, numPseudoparticles));

            float[] projectionAngles = new float[numPseudoparticles * 3];
            for (int k = 0; k < numPseudoparticles; k++)
            {
                projectionAngles[k] = angles[k].X;
                projectionAngles[k + 1] = angles[k].Y;
                projectionAngles[k + 2] = angles[k].Z;
            }
            float[] atomPositions = graph.GetAtomPositions();

            float[] realIntensities = graph.GetAtomIntensities();
            Random rand = new Random();
            float[] startIntensities = Helper.ArrayOfFunction(i => (float)(rand.NextDouble()), graph.Atoms.Count());
            AtomGraph graphBefore = new AtomGraph(graph);
            AtomGraph graphAfter = new AtomGraph(graph);
            graphBefore.SetAtomIntensities(startIntensities);

            IntPtr PseudoProjector = EntryPoint(PseudoProjectorParticles.GetHostContinuousCopy(), projectionAngles, inputVol.Dims, atomPositions, realIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());
            IntPtr PseudoReconstructor = EntryPoint(PseudoProjectorParticles.GetHostContinuousCopy(), projectionAngles, inputVol.Dims, atomPositions, startIntensities, graph.Atoms[0].R, (uint)graph.Atoms.Count());

            


            for (int k = 0; k < numPseudoparticles; k++)
            {
                GetProjection(PseudoProjector, PseudoProjectorParticles.GetHost(Intent.Write)[k], PseudoProjectorParticlesNorm.GetHost(Intent.Write)[k], angles[k] * Helper.ToDeg, 0.0f, 0.0f);
            }

            int numIt = 10;
            float[] itCorrs = new float[numIt+1];
            float[] err = new float[numIt];
            float[][] diff = Helper.ArrayOfFunction(i => new float[graph.Atoms.Count()], numIt);
            float[] newIntensities = new float[graph.Atoms.Count()];

            for (int l = 0; l < numPseudoparticles; l++)
            {
                GetProjection(PseudoReconstructor, PseudoProjectorCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
            }
            {
                float[] correlation = ImageProcessor.correlate(PseudoProjectorParticles, PseudoProjectorCurrProjections, ProjectorMasks);
                itCorrs[0] = correlation[0];
            }
            PseudoProjectorCurrProjections.WriteMRC($@"{outdir}\Projections_it0.mrc");

            for (int k = 0; k < numIt; k++)
            {
                err[k] = DoARTStep(PseudoReconstructor, PseudoProjectorParticles.GetHostContinuousCopy(), Helper.ArrayOfFunction(kx => angles[kx] * Helper.ToDeg, numPseudoparticles), 0.0f, 0.0f, (uint)numPseudoparticles);

                /* Get Current Projections */
                for (int l = 0; l < numPseudoparticles; l++)
                {
                    GetProjection(PseudoReconstructor, PseudoProjectorCurrProjections.GetHost(Intent.Write)[l], null, angles[l] * Helper.ToDeg, 0.0f, 0.0f);
                }
                float[] correlation = ImageProcessor.correlate(PseudoProjectorParticles, PseudoProjectorCurrProjections, ProjectorMasks);
                itCorrs[k + 1] = correlation[0];
                PseudoProjectorCurrProjections.WriteMRC($@"{outdir}\Projections_it{k+1}.mrc");

                GetIntensities(PseudoReconstructor, newIntensities);
                for (int idx = 0; idx < newIntensities.Count(); idx++)
                {
                    diff[k][idx] = newIntensities[idx] - realIntensities[idx];
                }

            }

            using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}itErr.txt"))
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

            using (System.IO.StreamWriter file = new System.IO.StreamWriter($@"{outdir}itCorrs.txt"))
            {
                file.WriteLine($"{itCorrs[0]}");
                for (int k = 0; k < numIt; k++)
                {
                    file.WriteLine($"it: {k}");
                    file.WriteLine($"{itCorrs[k + 1]}");
                }
            }




            graphAfter.SetAtomIntensities(newIntensities);

            graphBefore.Repr(1.0d).WriteMRC($"{outdir}GraphBefore.mrc");
            graphAfter.Repr(1.0d).WriteMRC($"{outdir}GraphAfter.mrc");

            PseudoProjectorParticles.WriteMRC($@"{outdir}\PseudoProjectorParticles.mrc");
            /* Create CTFs*/

            for (int idx = 0; idx < CTFParams.Length; idx++)
            {
                CTFParams[idx].Defocus = (decimal)(rand.NextDouble() + 1);
            }


            Image CTFCoords = CTF.GetCTFCoords(ProjectorParticles.Dims.X, ProjectorParticles.Dims.X);
            Image CTFs = new Image(ProjectorParticles.Dims, true);
            GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                            CTFCoords.GetDevice(Intent.Read),
                            (uint)CTFCoords.ElementsSliceComplex,
                            CTFParams.Select(p => p.ToStruct()).ToArray(),
                            false,
                            (uint)numParticles);




        }
    }
}
