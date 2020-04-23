using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using Warp.Headers;


namespace Testing
{
    class Program
    {
        static void Main(string[] args)
        {
            Image RefVol = Image.FromFile(@"D:\EMPIAR\10168\emd_4180_res7.mrc");
            Projector Ref = new Projector(RefVol, 3);


            Star starFile = new Star($@"D:\EMPIAR\10168\emd_4180_res7.projections_uniform.star");
            System.ValueTuple<string, int>[] micrographNames = starFile.GetRelionParticlePaths();
            Image micrograph = Image.FromFile($@"{micrographNames[0].Item1}");
            string name = micrographNames[0].Item1;
            Image[] particles = Helper.ArrayOfFunction(i =>
            {
                var item = micrographNames[i];
                if (item.Item1 != name)
                {
                    name = item.Item1;
                    micrograph = Image.FromFile($@"{name}");
                }
                return micrograph.AsSliceXY(item.Item2);

            }, micrographNames.Length);

            Image Particles = Image.Stack(particles);
            float3[] anglesDeg = starFile.GetRelionAngles();
            float3[] anglesRad = Helper.ArrayOfFunction(i=>anglesDeg[i] * Helper.ToRad, anglesDeg.Length);



            int NParticles = Particles.Dims.Z;
            Image RefProj = Ref.ProjectToRealspace(new int2(RefVol.Dims), anglesRad);
            RefProj.WriteMRC(@"D:\EMPIAR\10168\emd_4180_res7.projections_uniform.WARP_refParts.mrc", true);
            Image RefProjFT = RefProj.AsFFT();

            Image CTFs = new Image(RefProjFT.Dims, true);
            CTFs.TransformValues(f => 1.0f);

            
            RefProjFT.ShiftSlices(Helper.ArrayOfFunction(i => new float3(RefProj.Dims.X / 2, RefProj.Dims.Y / 2, 0), NParticles));
            Projector RefReconstructor = new Projector(new int3(RefProj.Dims.X), 2);
            RefReconstructor.BackProject(RefProjFT, CTFs, anglesDeg.Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));

            Image RefRec = RefReconstructor.ReconstructCPU(false, "C1");
            RefRec.WriteMRC(@"D:\EMPIAR\10168\emd_4180_res7.projections_uniform.WARP_refrecon.mrc", true);



            Image ParticlesFT = Particles.AsFFT();

            
               // Create weight 1 CTFs

            ParticlesFT.ShiftSlices(Helper.ArrayOfFunction(i => new float3(Particles.Dims.X / 2, Particles.Dims.Y / 2, 0), NParticles));

            Projector Reconstructor = new Projector(new int3(Particles.Dims.X), 2);

            Reconstructor.BackProject(ParticlesFT, CTFs, anglesDeg.Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));
            Image Rec = Reconstructor.ReconstructCPU(false, "C1");

            Rec.WriteMRC(@"D:\EMPIAR\10168\emd_4180_res7.projections_uniform.WARP_recon.mrc", true);

        }
    }
}
