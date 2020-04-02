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


        public static float[] correlate(Image A, Image B, Image mask)
        {
            float[] result = new float[A.Dims.Z];
            IntPtr d_result = GPU.MallocDeviceFromHost(result, A.Dims.Z);
            GPU.CorrelateRealspace(A.GetDevice(Intent.Read), B.GetDevice(Intent.Read), new int3(A.Dims.X, A.Dims.Y, 1), mask.GetDevice(Intent.Read), d_result, (uint)A.Dims.Z);
            GPU.CopyDeviceToHost(d_result, result, A.Dims.Z);
            return result;
                /*
            A = A.GetCopyGPU();
            B = B.GetCopyGPU();
            A.Normalize();
            B.Normalize();
            A.Multiply(B);
            float[] AElementsSlices = new float[A.Dims.Z];
            float[] AElementsSlicesInv = new float[A.Dims.Z];
            float[] BElementsSlices = new float[B.Dims.Z];
            float[] BElementsSlicesInv = new float[B.Dims.Z];
            if (mask != null)
            {
                A.Multiply(mask);
                B.Multiply(mask);
                Image maskSum = mask.AsSum2D();
                for (int i = 0; i < A.Dims.Z; i++)
                {
                    AElementsSlices[i] = maskSum.GetHost(Intent.Read)[i][0];
                    AElementsSlicesInv[i] = 1.0f / maskSum.GetHost(Intent.Read)[i][0];
                    BElementsSlices[i] = maskSum.GetHost(Intent.Read)[i][0];
                    BElementsSlicesInv[i] = 1.0f / maskSum.GetHost(Intent.Read)[i][0];
                }
            }
            else
            {
                for (int i = 0; i < A.Dims.Z; i++)
                {
                    AElementsSlices[i] = A.DimsSlice.Elements();
                    AElementsSlicesInv[i] = 1.0f / A.DimsSlice.Elements();
                    BElementsSlices[i] = B.DimsSlice.Elements();
                    BElementsSlicesInv[i] = 1.0f / B.DimsSlice.Elements();
                }
            }

            
            float[] AElementsSlices = new float[A.Dims.Z];
            float[] AElementsSlicesInv = new float[A.Dims.Z];
            float[] BElementsSlices = new float[B.Dims.Z];
            float[] BElementsSlicesInv = new float[B.Dims.Z];
            if (mask != null)
            {
                A.Multiply(mask);
                B.Multiply(mask);
                Image maskSum = mask.AsSum2D();
                for (int i = 0; i < A.Dims.Z; i++)
                {
                    AElementsSlices[i] = maskSum.GetHost(Intent.Read)[i][0];
                    AElementsSlicesInv[i] = 1.0f/maskSum.GetHost(Intent.Read)[i][0];
                    BElementsSlices[i] = maskSum.GetHost(Intent.Read)[i][0];
                    BElementsSlicesInv[i] = 1.0f/maskSum.GetHost(Intent.Read)[i][0];
                }
            }
            else
            {
                for (int i = 0; i < A.Dims.Z; i++)
                {
                    AElementsSlices[i] = A.DimsSlice.Elements();
                    AElementsSlicesInv[i] = 1.0f/A.DimsSlice.Elements();
                    BElementsSlices[i] = B.DimsSlice.Elements();
                    BElementsSlicesInv[i] = 1.0f/B.DimsSlice.Elements();
                }
            }

            Image A_mean = new Image(new int3(A.Dims.Z,1,1));

            GPU.Sum(A.GetDevice(Intent.Read), A_mean.GetDevice(Intent.Write), (uint)A.ElementsSliceReal, (uint)A.Dims.Z);

            Image A_mean_sqrd = A_mean.GetCopyGPU();
            A_mean_sqrd.Multiply(A_mean_sqrd); // (\Sum_i a_i)^2
            Image A_sqrd = A.GetCopyGPU();
            A_sqrd.Multiply(A_sqrd);    // (a_i)^2

            Image B_mean = B.AsSum2D();
            Image B_mean_sqrd = B_mean.GetCopyGPU();
            B_mean_sqrd.Multiply(B_mean_sqrd);
            Image B_sqrd = B.GetCopyGPU();
            B_sqrd.Multiply(B_sqrd);


            A_mean.Multiply(AElementsSlicesInv);    //\frac 1 n \Sum_i a_i
            B_mean.Multiply(BElementsSlicesInv);    //\frac 1 n \Sum_i b_i

            A.Add(-A_mean.GetHost(Intent.Read)[0][0]);     // a_i - \frac 1 n \Sum_i a_i
            B.Add(-B_mean.GetHost(Intent.Read)[0][0]);     // b_i - \frac 1 n \Sum_i b_i
            Image numerator = A.GetCopyGPU();
            numerator.Multiply(B);                          // (a_i - \frac 1 n \Sum_i a_i) (b_i - \frac 1 n \Sum_i b_i)
            Image t_numerator = numerator.AsSum2D();
            numerator.Dispose();
            numerator = t_numerator; // \Sum_i ((a_i - \frac 1 n \Sum_i a_i) (b_i - \frac 1 n \Sum_i b_i))
            numerator.Multiply(AElementsSlicesInv); // \frac 1 n \Sum_i ((a_i - \frac 1 n \Sum_i a_i) (b_i - \frac 1 n \Sum_i b_i))

            //denominator

            Image A_std = A_sqrd.AsSum2D();                 // \Sum_i (a_i)^2

            A_std.Multiply(AElementsSlices);              // n \Sum_i (a_i)^2

            A_std.Subtract(A_mean_sqrd);                    // n \Sum_i (a_i)^2- ( \Sum a_i )^2

            float tmp = A_std.GetHost(Intent.Read)[0][0];
            A_std.GetHost(Intent.Write)[0][0] = (float)Math.Sqrt(tmp);  // \sqrt{n \Sum_i (a_i)^2- ( \Sum a_i )^2}

            A_std.Multiply(AElementsSlicesInv);                   // \frac 1 n \sqrt{n \Sum_i (a_i)^2- ( \Sum a_i )^2} = \sqrt{\frac n n^2 \Sum_i (a_i)^2- ( \frac 1 n \Sum a_i )^2} = \sqrt{E[x^2]-(E[x])^2}


            Image B_std = B_sqrd.AsSum2D();

            B_std.Multiply(BElementsSlices);

            B_std.Subtract(B_mean_sqrd);

            tmp = B_std.GetHost(Intent.Read)[0][0];
            B_std.GetHost(Intent.Write)[0][0] = (float)Math.Sqrt(tmp);

            B_std.Multiply(BElementsSlicesInv);

            Image denominator = A_std.GetCopyGPU();
            denominator.Multiply(B_std);

            Image res_im = numerator.GetCopyGPU();
            res_im.Divide(denominator);

            float[] result = res_im.GetHostContinuousCopy();
            A.Dispose();
            B.Dispose();

            A_mean.Dispose();
            A_mean_sqrd.Dispose();
            A_sqrd.Dispose();
            B_mean.Dispose();
            B_mean_sqrd.Dispose();
            B_sqrd.Dispose();
            numerator.Dispose();
            A_std.Dispose();
            B_std.Dispose();
            denominator.Dispose();
            res_im.Dispose();
            return result;*/
        }

        public static void Normalize01(Image im)
        {

            double min = double.MaxValue, max = double.MinValue;
            double sum = 0;
            im.TransformValues(f =>
            {
                if (f < min)
                    min = f;
                if (f > max)
                    max = f;
                return f;
            });
            im.Add((float)-min);
            //double mean = im.AsSum3D().GetHost(Intent.Read)[0][0];
            im.Multiply((float)(1.0f/(max-min)));
        }

        public static Image Downsample(Image im, float factor)
        {
            int3 oldSize = im.Dims;
            float falloff = 10.0f;
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
            Cosine.WriteMRC($@"D:\EMPIAR\10168\Cosine.mrc");
            ft.WriteMRC($@"D:\EMPIAR\10168\ft_before_mult.mrc");
            ft.Multiply(Cosine);
            ft.WriteMRC($@"D:\EMPIAR\10168\ft_after_mult.mrc");
            ft = ft.AsPadded(newSize);
            ft.WriteMRC($@"D:\EMPIAR\10168\ft_padded.mrc");
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
