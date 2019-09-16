using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp.Tools;
using Warp;

namespace FlexibleRefinement.Util
{
    public class ImageProcessor
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


        public static Image Downsample(Image im, float factor)
        {
            int3 oldSize = im.Dims;
            float falloff = 5.0f;
            int3 newSize = oldSize / factor;
            float innerRadius = (newSize - newSize / 2).Length() - (1.1f * falloff);
            Image ft = im.AsFFT(true);
            Image Cosine = new Image(ft.Dims, true);
            float[][] CosineData = Cosine.GetHost(Intent.Write);
            double CosineSum = 0;
            for (int z = 0; z < Cosine.DimsFT.Z; z++)
            {
                int zz = z;
                if (z > Cosine.DimsFT.Z / 2)
                {
                    zz = Cosine.DimsFT.Z - z;
                }
                zz *= zz;
                for (int y = 0; y < Cosine.DimsFT.Y; y++)
                {
                    int yy = y;
                    if (y > Cosine.DimsFT.Y / 2)
                    {
                        yy = Cosine.DimsFT.Y - y;
                    }

                    yy *= yy;
                    for (int x = 0; x < Cosine.DimsFT.X; x++)
                    {
                        int xx = x;
                        xx *= xx;

                        float R = (float)Math.Sqrt(xx + yy + zz);
                        double C = Math.Cos(Math.Max(0, Math.Min(falloff, R - innerRadius)) / falloff * Math.PI) * 0.5 + 0.5;

                        CosineSum += C;
                        CosineData[z][y * Cosine.DimsFT.X + x] = (float)C;
                    }
                }
            }

            ft.Multiply(Cosine);

            ft = ft.AsPadded(newSize);
            Image newIm = ft.AsIFFT(true);
            GPU.Normalize(newIm.GetDevice(Intent.Read),
                         newIm.GetDevice(Intent.Write),
                         (uint)newIm.ElementsReal,
                         (uint)1);

            return newIm;

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
