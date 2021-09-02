using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using Warp.Headers;
using System.IO;

namespace Testing
{
    class Program
    {
        static void Main(string[] args)
        {
            String outdir = args[1];
            Star starFile = new Star(args[0]);
            string instarName = Path.GetFileName(args[0].Replace(".star", ""));
            string starDir = Path.GetDirectoryName(args[0]);

            System.ValueTuple<string, int>[] micrographNames = starFile.GetRelionParticlePaths();
            Image micrograph = Image.FromFile($@"{starDir}\{micrographNames[0].Item1}");
            //micrograph.WriteMRC("micrograph.mrc", true);
            string name = micrographNames[0].Item1;
            Image[] particles = Helper.ArrayOfFunction(i =>
            {
                var item = micrographNames[i];
                if (item.Item1 != name)
                {
                    name = item.Item1;
                    micrograph = Image.FromFile($@"{starDir}\{name}");
                }
                return micrograph.AsSliceXY(item.Item2);

            }, micrographNames.Length);

            Image Particles = Image.Stack(particles);
            Particles.FreeDevice();
           // Particles.WriteMRC("Particles.mrc", true);
            //particles[0].WriteMRC("particles_0.mrc", true);

            float3[] anglesDeg = starFile.GetRelionAngles();
            float3[] anglesRad = Helper.ArrayOfFunction(i => anglesDeg[i] * Helper.ToRad, anglesDeg.Length);

            int NParticles = Particles.Dims.Z;
            {
                Image CTFIm = new Image(Particles.Dims, true);
                CTFIm.TransformValues(f => 1.0f);
                Image[] CTFs = Helper.ArrayOfFunction(i => CTFIm.AsSliceXY(i), CTFIm.Dims.Z);

                Projector Reconstructor = new Projector(new int3(Particles.Dims.X), 2);

                for (int processIdx = 0; processIdx < particles.Length; processIdx += 1024)
                {

                    Image part = Image.Stack(particles.Skip(processIdx).Take(1024).ToArray());
                    //part.WriteMRC($"{processIdx}.mrc", true);
                    Image ft = part.AsFFT();
                    Image partCTF = Image.Stack(CTFs.Skip(processIdx).Take(1024).ToArray());
                    ft.ShiftSlices(Helper.ArrayOfFunction(j => new float3(part.Dims.X / 2, part.Dims.Y / 2, 0), part.Dims.Z));
                    Reconstructor.BackProject(ft, partCTF, anglesDeg.Skip(processIdx).Take(part.Dims.Z).Select(a => a * Helper.ToRad).ToArray(), new float3(1, 1, 0));
                    part.Dispose();
                    partCTF.Dispose();
                    ft.Dispose();
                }
                Image Rec = Reconstructor.Reconstruct(false, "C1");
                Rec.WriteMRC($@"{outdir}\{instarName}.WARP_recon.mrc", true);
            }
        }
    }
}
