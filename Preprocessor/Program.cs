using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Warp;
using Warp.Tools;
using FlexibleRefinement.Util;
using System.IO;
using Warp.Headers;

namespace Preprocessor
{
    class Program
    {
        static void Main(string[] args)
        {
            // The code provided will print ‘Hello World’ to the console.
            // Press Ctrl+F5 (or go to Debug > Start Without Debugging) to run your app.


            // Go to http://aka.ms/dotnet-get-started-console to continue learning how to build a console app!


            float targetRes = 7; // 7 Angstroms
            float3 newPix = new float3(targetRes / 2, targetRes / 2, targetRes / 2);
            string inputVol = @"D:\EMPIAR\10168\emd_4180.mrc";
            string inputMask = @"D:\EMPIAR\10168\emd_4180.mask.mrc";
            string outputVol = $@"D:\EMPIAR\10168\emd_4180_res{targetRes}.mrc";
            string outputMask = $@"D:\EMPIAR\10168\emd_4180_res{targetRes}.mask.mrc";

            Image inIm = Image.FromFile(inputVol);
            Image inMask = Image.FromFile(inputMask);
            HeaderMRC Header;
            using (BinaryReader Reader = new BinaryReader(File.OpenRead(inputVol)))
            {
                Header = new HeaderMRC(Reader);

            }
            float3 olPix = Header.PixelSize;
            float factor = (float)(newPix.X / olPix.X);

            Image outIm = ImageProcessor.Downsample(inIm, factor);
            outIm.WriteMRC(outputVol);
            Image outMask = inMask.AsScaled(outIm.Dims);
            outMask.Binarize(0.1f);
            outMask.WriteMRC(outputMask);


        }
    }
}
