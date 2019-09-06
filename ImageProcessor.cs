using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;

namespace FlexibleRefinement
{
    class ImageProcessor
    {
        public static float3[][] getGradient(Image im)
        {
            float3[][] grad = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new float3(0), im.Dims.X*im.Dims.Y), im.Dims.Z);
            float[][] dataIn = im.GetHost(Intent.Read);

            Helper.ForCPU(0, im.Dims.Z, 24, null, (z, id, ts) =>
             {
                 for (int y = 0; y < im.Dims.Y; y++)
                 {
                     for (int x = 0; x < im.Dims.X; x++)
                     {
                         float localVal = dataIn[z][y * im.Dims.X + x];
                         float gradX = ((x + 1) < im.Dims.X ? (dataIn[z][y * im.Dims.X + (x + 1)] - localVal) : 0) - ((x - 1) >= 0 ? (dataIn[z][y * im.Dims.X + (x - 1)] - localVal) : 0);
                         float gradY = ((y + 1) < im.Dims.Y ? (dataIn[z][(y + 1) * im.Dims.X + x] - localVal) : 0) - ((y - 1) >= 0 ? (dataIn[z][(y - 1) * im.Dims.X + x] - localVal) : 0);
                         float gradZ = ((z + 1) < im.Dims.Z ? (dataIn[z + 1][y * im.Dims.X + x] - localVal) : 0) - ((z - 1) >= 0 ? (dataIn[z - 1][y * im.Dims.X + x] - localVal) : 0);
                         grad[z][y * im.Dims.X + x] = new float3(gradX, gradY, gradZ);
                     }

                 }
             }, null);
            return grad;
        }


        public static float3[] FillWithEquidistantPoints(Image mask, int n, out float R, float r0 = 0.0f)
        {
            float3 MaskCenter = mask.AsCenterOfMass();
            float[] MaskData = mask.GetHostContinuousCopy();
            int3 Dims = mask.Dims;

            float3[] BestSolution = null;

            float a = 0, b = Dims.X / 2;
            if (r0 > 0.0f)
            {
                R = r0;
            }
            else
            {
                R = (a + b) / 2;
            }
            float3 Offset = new float3(0, 0, 0);

            for (int o = 0; o < 2; o++)
            {
                for (int i = 0; i < 10; i++)
                {
                    R = (a + b) / 2;

                    float Root3 = (float)Math.Sqrt(3);
                    float ZTerm = (float)(2 * Math.Sqrt(6) / 3);
                    float SpacingX = R * 2;
                    float SpacingY = Root3 * R;
                    float SpacingZ = ZTerm * R;
                    int3 DimsSphere = new int3(Math.Min(512, (int)Math.Ceiling(Dims.X / SpacingX)),
                                               Math.Min(512, (int)Math.Ceiling(Dims.Y / SpacingX)),
                                               Math.Min(512, (int)Math.Ceiling(Dims.Z / SpacingX)));
                    BestSolution = new float3[DimsSphere.Elements()];

                    for (int z = 0; z < DimsSphere.Z; z++)
                    {
                        for (int y = 0; y < DimsSphere.Y; y++)
                        {
                            for (int x = 0; x < DimsSphere.X; x++)
                            {
                                BestSolution[DimsSphere.ElementFromPosition(x, y, z)] = new float3(2 * x + (y + z) % 2,
                                                                                                   Root3 * (y + 1 / 3f * (z % 2)),
                                                                                                   ZTerm * z) * R + Offset;
                            }
                        }
                    }

                    List<float3> InsideMask = BestSolution.Where(p =>
                    {
                        int3 ip = new int3(p);
                        if (ip.X >= 0 && ip.X < Dims.X && ip.Y >= 0 && ip.Y < Dims.Y && ip.Z >= 0 && ip.Z < Dims.Z)
                            return MaskData[Dims.ElementFromPosition(new int3(p))] == 1;
                        return false;
                    }).ToList();
                    BestSolution = InsideMask.ToArray();

                    if (BestSolution.Length == n)
                        break;
                    else if (BestSolution.Length < n)
                        b = R;
                    else
                        a = R;
                }

                float3 CenterOfPoints = MathHelper.Mean(BestSolution);
                Offset = MaskCenter - CenterOfPoints;

                a = 0.8f * R;
                b = 1.2f * R;
            }

            BestSolution = BestSolution.Select(v => v + Offset).ToArray();

            return BestSolution;
        }
    }
}
