using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;

namespace GaussProjection
{
    class Program
    {
        static void Main(string[] args)
        {
            reconstructionTest2();
        }

        static void gaussProjectTest()
        {
            int3 dims = new int3(100);
            Image Gauss3D = new Image(dims);

            double sigmaSq = 25;
            float3 mu = new float3(48.3f, 49.2f, 32.2f);
            Gauss3D.TransformValues((x, y, z, value) =>
            (float)Math.Exp(-(new float3(x, y, z) - mu).LengthSq() / sigmaSq)
            );

            Gauss3D.WriteMRC("Gauss3D.mrc");
            Projector proj = new Projector(Gauss3D, 2);

            Image Gauss3Dproj = proj.ProjectToRealspace(new int2(100), new float3[] { new float3(0, (float)Math.PI / 2, 0) });
            Gauss3Dproj.WriteMRC("Gauss3Dproj.mrc");

        }

        public static float3 VecToRot(float4 vec)
        {
            float phi = 0.0f;
            if (vec.X == 0 )
            {
                if (vec.Y == 0)
                {
                    phi = 0.0f;
                }
                else
                {
                    phi = (float) (Math.PI / 2);
                }
            }
            
            float theta = (float)(Math.Acos(vec.Z));
            float psi = vec.W - phi;

            return new float3(phi, theta, psi);
        }

        static void reconstructionTest2()
        {
            Star TableIn = new Star(@"D:\florian_debug\particles.star");
            CTF[] CTFParams = TableIn.GetRelionCTF();

            
            Image inputVol = Image.FromFile($@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy_Modulated\100\startIm.mrc");
            Projector proj = new Projector(inputVol, 2);
            
            int numParticles = 300;
            int numAngles = numParticles;
            CTFParams.Take(numParticles).ToArray();

            int numAnglesX = (int)Math.Ceiling(Math.Pow(numAngles, 1.0d/3.0d));
            int numAnglesY = (int)Math.Max(1,Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            int numAnglesZ = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            numAngles = numAnglesX * numAnglesY * numAnglesZ;
            numParticles = numAngles;
            float3[] angles = new float3[numAngles];
            int i = 0;
            for (int x=0; x< numAnglesX; x++)
            {
                float xx = (float)(2*Math.PI) / (numAnglesX - 1) * (x) ;
                for (int y = 0; y < numAnglesY; y++)
                {
                    float yy = (float)(2 * Math.PI) / (numAnglesY - 1) * (y) ;
                    for (int z = 0; z < numAnglesZ; z++)
                    {
                        float zz = (float)(2 * Math.PI) / (numAnglesZ - 1) * (z) ;
                        angles[i] = new float3(xx, yy, zz);
                        i++;
                    }
                }
            }
            


            
            Image Particles = proj.ProjectToRealspace(new int2(100), angles);

            GPU.Normalize(Particles.GetDevice(Intent.Read),
                            Particles.GetDevice(Intent.Write),
                            (uint)Particles.ElementsSliceReal,
                            (uint)angles.Count());

            Particles.WriteMRC("Particles.mrc");

            /* Create CTFs*/

            Image CTFCoords = CTF.GetCTFCoords(Particles.Dims.X, Particles.Dims.X);
            Image CTFs = new Image(Particles.Dims, true);
            GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                            CTFCoords.GetDevice(Intent.Read),
                            (uint)CTFCoords.ElementsSliceComplex,
                            CTFParams.Select(p => p.ToStruct()).ToArray(),
                            false,
                            (uint)numParticles);


            

            Image ParticlesFt = Particles.AsFFT();
            ParticlesFt.Multiply(CTFs); //Simulate CTF influence
            Image ParticlesWeight = new Image(ParticlesFt.Dims, true);
            ParticlesWeight.TransformValues((x, y, z, value) => 1.0f);

            /* Possibility 1, weighted back projection with CTF */
            Projector Reconstructor1 = new Projector(new int3(Particles.Dims.X), 2);

            ParticlesFt.ShiftSlices(Helper.ArrayOfFunction(j => new float3(Particles.Dims.X / 2, Particles.Dims.Y / 2, 0), numParticles));
            
            ParticlesWeight.WriteMRC("Weights.mrc");
            Reconstructor1.BackProject(ParticlesFt, CTFs, angles, new float3(1,1,0));

            Image Rec1 = Reconstructor1.ReconstructCPU(false, "C1");

            Rec1.WriteMRC(@"rec1.mrc", true);


            /* Possibility 2, weighted back projection without CTF */
            Projector Reconstructor2 = new Projector(new int3(Particles.Dims.X), 2);

            ParticlesFt.ShiftSlices(Helper.ArrayOfFunction(j => new float3(Particles.Dims.X / 2, Particles.Dims.Y / 2, 0), numParticles));
            

            ParticlesWeight.WriteMRC("Weights.mrc");
            Reconstructor2.BackProject(ParticlesFt, ParticlesWeight, angles, new float3(1, 1, 0));

            Image Rec2 = Reconstructor2.ReconstructCPU(false, "C1");

            Image CTFSum = new Image(new int3(CTFs.Dims.X, CTFs.Dims.Y, 1), true);
            float[][] CTFsData = CTFs.GetHost(Intent.Read);

            CTFSum.TransformValues((x, y, z, value) =>
            {
                float sum = 0;
                for (z = 0; z < CTFs.Dims.Z; z++)
                {
                    sum += CTFsData[z][y * CTFs.DimsFT.X + x];
                }
                return sum;
            });

            CTFSum.WriteMRC("CTFSum.mrc");
            Rec2.WriteMRC(@"rec2.mrc", true);
        }

        static void reconstructionTest()
        {
            // Get metadata

            Star TableIn = new Star(@"D:\florian_debug\particles.star");
            float3[] Angles = TableIn.GetRelionAngles();
            float3[] Offsets = TableIn.GetRelionOffsets();
            CTF[] CTFParams = TableIn.GetRelionCTF();

            // Load particles and scale them to 3.8/2 A/px, then shift by offsets from STAR file

            Image Particles = Image.FromFile(@"D:\florian_debug\FoilHole_14503666_Data_14508721_14508722_20181214_2318_Fractions.mrcs");
            int NParticles = Particles.Dims.Z;
            int Dim = Particles.Dims.X;
            int DimScaled = (int)((float)CTFParams[0].PixelSize * 2 / 3.8f * Dim / 2) * 2;
            float ScaleFactor = (float)DimScaled / Dim;

            for (int i = 0; i < CTFParams.Length; i++)
                CTFParams[i].PixelSize = 3.8M / 2;

            Particles = Particles.AsScaled(new int2(DimScaled));
            Particles.ShiftSlices(Helper.ArrayOfFunction(i => Offsets[i] * ScaleFactor, NParticles));

            // Simulate CTFs and take the one for particle 0

            Image CTFCoords = CTF.GetCTFCoords(Particles.Dims.X, Particles.Dims.X);
            Image CTFs = new Image(Particles.Dims, true);
            GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                            CTFCoords.GetDevice(Intent.Read),
                            (uint)CTFCoords.ElementsSliceComplex,
                            CTFParams.Select(p => p.ToStruct()).ToArray(),
                            false,
                            (uint)NParticles);
            Image CTF0 = new Image(CTFs.GetHost(Intent.Read)[0], new int3(DimScaled, DimScaled, 1), true);

            // Perform beam tilt correction, though it won't change much at 3.8 A resolution

            Image BeamTiltCorr = Image.Stack(CTFParams.Select(v => v.GetBeamTilt(Particles.Dims.X, Particles.Dims.X)).ToArray());

            Image ParticlesFT = Particles.AsFFT();
            ParticlesFT.Multiply(BeamTiltCorr);
            Particles = ParticlesFT.AsIFFT();

            // Normalize particles and take particle 0

            GPU.Normalize(Particles.GetDevice(Intent.Read),
                            Particles.GetDevice(Intent.Write),
                            (uint)Particles.ElementsSliceReal,
                            (uint)NParticles);

            Image Part0 = new Image(Particles.GetHost(Intent.Read)[0], new int3(DimScaled, DimScaled, 1));
            Part0.WriteMRC(@"D:\florian_debug\part0.mrc", true);

            // Load reference volume, scale to 3.8/2 A/px

            Image RefVol = Image.FromFile(@"D:\florian_debug\run_class001_cropped_220_220_220.mrc");
            RefVol = RefVol.AsScaled(new int3(DimScaled));
            Projector Ref = new Projector(RefVol, 3);

            // Make projections around particle 0's angles and multiply by CTF, then normalize

            Image RefProj0 = Ref.ProjectToRealspace(new int2(DimScaled), Helper.ArrayOfFunction(i => (Angles[0] + new float3(0, 0, (i - 50) / 10f)) * Helper.ToRad, 101));
            Image RefProj0FT = RefProj0.AsFFT();
            RefProj0FT.MultiplySlices(CTF0);
            RefProj0 = RefProj0FT.AsIFFT();

            GPU.Normalize(RefProj0.GetDevice(Intent.Read),
                            RefProj0.GetDevice(Intent.Write),
                            (uint)RefProj0.ElementsSliceReal,
                            (uint)RefProj0.Dims.Z);
            RefProj0.WriteMRC(@"D:\florian_debug\proj.mrc", true);

            // Multiply projections with particle 0 and calculate scores

            RefProj0.MultiplySlices(Part0);
            float[] Scores = RefProj0.GetHost(Intent.Read).Select(v => MathHelper.Mean(v)).ToArray();

            // To make sure everything had the correct orientations, make a quick reconstruction from all particles
            // Decenter first since we will be rotating them in Fourier space

            ParticlesFT = Particles.AsFFT();
            ParticlesFT.ShiftSlices(Helper.ArrayOfFunction(i => new float3(Particles.Dims.X / 2, Particles.Dims.Y / 2, 0), NParticles));

            // Multiply by CTF, and take CTF^2 for weighting to deconvolve the reconstruction

            ParticlesFT.Multiply(BeamTiltCorr);
            ParticlesFT.Multiply(CTFs);
            CTFs.Multiply(CTFs);

            Projector Reconstructor = new Projector(new int3(Particles.Dims.X), 1);

            // Back-project and reconstruct with octahedral symmetry

            Reconstructor.BackProject(ParticlesFT, CTFs, Angles.Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));

            Image Rec = Reconstructor.ReconstructCPU(false, "O");

            Rec.WriteMRC(@"D:\florian_debug\rec.mrc", true);
        }
    }
}
