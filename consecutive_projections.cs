using System;
using System.Collections.Generic;
using System.Globalization;
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
            /*Image refVol = Image.FromFile(@"D:\FlexibleRefinementResults\input\input2\emd_9233_Scaled_2.0.mrc");
            var angles = Helper.GetHealpixAngles(1);

            Projector proj = new Projector(refVol, 2);
            Image projections = proj.ProjectToRealspace(refVol.DimsSlice, angles);
            projections.WriteMRC(@"D:\FlexibleRefinementResults\Results\Consecutive_Rastering\Output\refProjections_cs.mrc", true);
            */
            string refFileName = args[0];
            // string refMaskName = args[1];
            string atomProjectionsName = args[1];

            Image RefProjections = Image.FromFile(refFileName);
            Image RefProjectionsMask = new Image(RefProjections.Dims);
            RefProjectionsMask.Fill(1);
            RefProjectionsMask.MaskSpherically(47, 4, false);

            RefProjections.Multiply(RefProjectionsMask);
            Image RefProjectionsFT = RefProjections.AsFFT();
            float[][] RefProjectionsFTData = RefProjectionsFT.GetHost(Intent.Read);



            Image AtomProjections = Image.FromFile(atomProjectionsName);
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

                string[] FRC = Shells.Select(v => ( v.X / (float)Math.Max(1e-16, Math.Sqrt(v.Y * v.Z))).ToString(CultureInfo.InvariantCulture)).ToArray();
                string[] idxs = Helper.ArrayOfFunction(k => $"{k}", FRC.Length);
                string[][] colums = { idxs,FRC };
                string[] names = { "rlnSpectralIndex", "rlnFourierShell" };

                new Star(colums, names).Save($@"{atomProjectionsName.Replace(".mrc", "")}_frc_vs_ref_masked.star");
            }
        }

    }
}


