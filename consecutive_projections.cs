using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
namespace Consecutive_Rastering_Projections
{
    class consecutive_projections
    {

        static void Main(string[] args)
        {
            int[] N_series = new int[11] { 1000, 10000, 20000, 30000, 50000, 100000, 200000, 500000, 600000, 800000, 1000000 };
            int[] Super_series = new int[3] { 1, 2, 4 };


            Image RefFTProjections = Image.FromFile(@"D:\EMD\9233\Projections_2.0_uniform\projections_uniform.mrc");
            RefFTProjections.MaskSpherically(RefFTProjections.Dims.X / 2, 4, false);
            Image RefProjectionsFT = RefFTProjections.AsFFT();
            float[][] RefProjectionsFTData = RefProjectionsFT.GetHost(Intent.Read);
            foreach (var N in N_series)
            {
                foreach (var super in Super_series)
                {
                    string outdir = $@"D:\EMD\9233\Consecutive_Rastering\weighting_true_{super}.000000_{N/1000}";
                    for (int idx = 0; idx < 10; idx++)
                    {

                        Image AtomProjections = Image.FromFile($@"{outdir}\{idx}_proj.mrc");
                        Image AtomProjectionsFT = AtomProjections.AsFFT();
                        float[][] AtomProjectionsFTData = AtomProjectionsFT.GetHost(Intent.Read);


                        float3[] Shells = new float3[AtomProjections.Dims.X / 2];
                        for (int a = 0; a < AtomProjections.Dims.Z; a++)
                        {
                            float[] AData = AtomProjectionsFTData[a];
                            float[] RData = RefProjectionsFTData[a];

                            int i = 0;
                            Helper.ForEachElementFT(new int2(AtomProjections.Dims.X), (x, y, xx, yy, r, angle) =>
                            {
                                int R = (int)Math.Round(r);
                                if (R >= Shells.Length)
                                    return;

                                float2 A = new float2(AData[i * 2], AData[i * 2 + 1]);
                                float2 B = new float2(RData[i * 2], RData[i * 2 + 1]);

                                float AB = A.X * B.X + A.Y * B.Y;
                                float A2 = A.LengthSq();
                                float B2 = B.LengthSq();

                                Shells[R] += new float3(AB, A2, B2);

                                i++;
                            });
                        }

                        float[] FRC = Shells.Select(v => v.X / (float)Math.Max(1e-16, Math.Sqrt(v.Y * v.Z))).ToArray();
                        new Star(FRC, "frc").Save($@"{outdir}\{idx}_frc_vs_ref_masked.star");


                        AtomProjections.Dispose();
                        AtomProjectionsFT.Dispose();

                    }



                }
            }
            RefFTProjections.Dispose();
            RefProjectionsFT.Dispose();
        }
    }
}


