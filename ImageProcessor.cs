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
    }
}
