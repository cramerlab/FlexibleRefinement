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

        static void downsampleToDim()
        {
            string inputVol = @"D:\EMPIAR\10168\emd_4180_res7_10k_conv_it3_oversampled2.mrc";
            int3 targetDim = new int3(92);
            Image inp = Image.FromFile(inputVol);
            Image fft = inp.AsFFT(true);
            Image paddedfft = fft.AsPadded(targetDim);
            Image outp = paddedfft.AsIFFT();

            outp.WriteMRC(@"D:\EMPIAR\10168\emd_4180_res7_10k_conv_it3_oversampled2_downsampled.mrc");


        }

        static void downsampler()
        {
            //float targetRes = 7.0f; // 7 Angstroms
            float targetRes = 3.5f; // 3.5 Angstroms
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
            
            Image outMask = inMask.AsScaled(outIm.Dims);
            outMask.Binarize(0.1f);
            outMask.WriteMRC(outputMask, true);
            outIm.Normalize();
            outIm.Multiply(outMask);
            outIm.WriteMRC(outputVol, true);
        }


        static void projectUniform(string inputVolPath = @"D:\EMPIAR\10168\emd_4180_res7.mrc", string starPath = @"D:\EMPIAR\10168\shiny.star", string outdir = @"D:\EMPIAR\10168\", string projDir = @"D:\EMPIAR\10168\Projections_7_uniform", int numParticles = 500)
        {
            int batchSize = 1024;
            numParticles = 10*batchSize;


            Image inVol = Image.FromFile(inputVolPath);
            int2 projDim = new int2(inVol.Dims.X);
            Projector proj = new Projector(inVol, 3);
            
            

            if (!Directory.Exists(projDir))
            {
                Directory.CreateDirectory(projDir);
            }

            
            Star starInFile = new Star(starPath, "particles");
            Star starCleanOutFile = new Star(starInFile.GetColumnNames());
            Star starConvolvedOutFile = new Star(starInFile.GetColumnNames());
            Star starConvolved2OutFile = new Star(starInFile.GetColumnNames());


            CTF[] CTFParams = starInFile.GetRelionCTF();
            Image CTFCoords = CTF.GetCTFCoords(projDim, projDim, 3.0f);

            

            int numAngles = numParticles;
            int numAnglesX = (int)Math.Ceiling(Math.Pow(numAngles, 1.0d / 3.0d));
            int numAnglesY = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            int numAnglesZ = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 3.0d)));
            numAngles = numAnglesX * numAnglesY * numAnglesZ;
            numParticles = numAngles;

            float3[] anglesRad = new float3[numAngles];
            {
                int i = 0;
                for (int x = 0; x < numAnglesX; x++)
                {
                    float xx = (float)(2 * Math.PI) / (numAnglesX - 1) * (x);
                    for (int y = 0; y < numAnglesY; y++)
                    {
                        float yy = (float)(2 * Math.PI) / (numAnglesY - 1) * (y);
                        for (int z = 0; z < numAnglesZ; z++)
                        {
                            float zz = (float)(2 * Math.PI) / (numAnglesZ - 1) * (z);
                            anglesRad[i] = new float3(xx, yy, zz);
                            i++;
                        }
                    }
                }
            }


            float3[] anglesDeg = Helper.ArrayOfFunction(i => anglesRad[i] * Helper.ToDeg, anglesRad.Count());

            List<List<string>> rows = starInFile.GetAllRows();
            int n = 0;
            Image CTFSum = new Image(new int3(projDim), true);
            for (int i = 0; i < Math.Min(10, numAngles / batchSize); i++)
            {
                float3[] partAnglesRad = anglesRad.Skip(((i) * batchSize)).Take(batchSize).ToArray();

                float3[] partAnglesDeg = Helper.ArrayOfFunction(k => partAnglesRad[k] * Helper.ToDeg, partAnglesRad.Count());
                Image im = proj.ProjectToRealspace(projDim, partAnglesRad);
                Image CTFs = new Image(im.Dims, true);
                GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                    CTFCoords.GetDevice(Intent.Read),
                    (uint)CTFCoords.ElementsSliceComplex,
                    CTFParams.Skip(((i) * batchSize)).Select(p => p.ToStruct()).ToArray(),
                    false,
                    (uint)im.Dims.Z);
                im.WriteMRC($@"{projDir}\{i}.mrc", true);
                CTFs.WriteMRC($@"{projDir}\{i}_ctf.mrc", true);

                /* convolve once */
                Image imFFT = im.AsFFT();
                im.Dispose();
                imFFT.Multiply(CTFs);
                im = imFFT.AsIFFT();
                imFFT.Dispose();
                im.WriteMRC($@"{projDir}\{i}_convolved.mrc", true);

                /* convolve with ctf^2 */
                imFFT = im.AsFFT();
                im.Dispose();
                imFFT.Multiply(CTFs);
                im = imFFT.AsIFFT();
                imFFT.Dispose();
                im.WriteMRC($@"{projDir}\{i}_convolved2.mrc", true);
                CTFs.Dispose();
                for (int j = 0; j < batchSize; j++)
                {
                    List<string> row = rows[n];
                    row[3] = $@"{j + 1}@{projDir}\{i}.mrc";
                    row[12] = partAnglesDeg[j].X.ToString();
                    row[13] = partAnglesDeg[j].Y.ToString();
                    row[14] = partAnglesDeg[j].Z.ToString();

                    starCleanOutFile.AddRow(new List<string>(row));

                    row[3] = $@"{j + 1}@{projDir}\{i}_convolved.mrc";
                    starConvolvedOutFile.AddRow(new List<string>(row));

                    row[3] = $@"{j + 1}@{projDir}\{i}_convolved2.mrc";
                    starConvolved2OutFile.AddRow(new List<string>(row));
                    n++;
                }
                
            }
            starCleanOutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projections_uniform.star");
            starConvolvedOutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projectionsConv_uniform.star");
            starConvolved2OutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projectionsConv2_uniform.star");
        }
        static void projectorByStar()
        {
            string inputVolPath = $@"D:\EMPIAR\10168\emd_4180_res7.mrc";
            Image inVol = Image.FromFile(inputVolPath);
            Projector proj = new Projector(inVol, 3);
            string outdir = $@"D:\EMPIAR\10168\";
            string projDir = $@"{outdir}\Projections_7"; 
            if (!Directory.Exists(projDir))
            {
                Directory.CreateDirectory(projDir);
            }
            int2 projDim = new int2(inVol.Dims.X);
            int batchSize = 1024;
            string starPath = $@"D:\EMPIAR\10299\data\Particles\shiny.star";
            Star starInFile = new Star(starPath, "particles");
            Star starOutFile = new Star(starInFile.GetColumnNames());
            List<List<string>> rows = starInFile.GetAllRows();
            float3[] anglesDeg = starInFile.GetRelionAngles();
            int numAngles = anglesDeg.Count();
            float3[] anglesRad = Helper.ArrayOfFunction(i => anglesDeg[i]*Helper.ToRad, numAngles);
            int n = 0;
            for (int i = 0; i < Math.Min(10, numAngles /batchSize); i++)
            {
                Image im = proj.ProjectToRealspace(projDim, anglesRad.Skip((i - 1 * batchSize)).Take(batchSize).ToArray());
                im.WriteMRC($@"{projDir}\{i}.mrc", true);
                for (int j = 0; j < batchSize; j++)
                {
                    List<string> row = rows[n];
                    row[3] = $@"{j+1}@{projDir}\{i}.mrc";
                    starOutFile.AddRow(row);
                    n++;
                }
            }
            starOutFile.Save($@"{inputVolPath.Replace(".mrc","")}.projections.star");
            
        }

        static void preprocess_emd_9233()
        {

            Image inVol = Image.FromFile($@"D:\EMD\9233\emd_9233.mrc");
            HeaderMRC Header;
            using (BinaryReader Reader = new BinaryReader(File.OpenRead($@"D:\EMD\9233\emd_9233.mrc")))
            {
                Header = new HeaderMRC(Reader);

            }
            float3 olPix = Header.PixelSize;
            float3 newPix = new float3(1.5f);
            Image sampledVol = ImageProcessor.Downsample(inVol, newPix.X / olPix.X);
            Image scaledVol = inVol.AsScaled(sampledVol.Dims);
            Image scaledMask = scaledVol.GetCopy();
            scaledMask.Binarize(0.01f);
            scaledMask = scaledMask.AsConvolvedGaussian(2.0f);
            scaledMask.Binarize(0.2f);
            scaledMask = scaledMask.AsConvolvedGaussian(2.0f);
            scaledMask.Binarize(0.2f);
            scaledMask = scaledMask.AsConvolvedGaussian(5.0f);
            scaledMask.Binarize(0.2f);
            scaledMask.WriteMRC($@"D:\EMD\9233\emd_9233_Scaled_1.5_mask.mrc");
            scaledVol.Multiply(scaledMask);
            scaledVol.WriteMRC($@"D:\EMD\9233\emd_9233_Scaled_1.5.mrc");
            //inVol.Normalize();
            //inVol.WriteMRC($@"D:\EMD\9233\emd_9233_normalized.mrc", true, Header);
            inVol.Binarize(0.01f);
            inVol = inVol.AsConvolvedGaussian(2.0f);
            inVol.Binarize(0.2f);
            inVol = inVol.AsConvolvedGaussian(2.0f);
            inVol.Binarize(0.2f);
            inVol = inVol.AsConvolvedGaussian(10.0f);
            inVol.Binarize(0.2f);
            inVol.WriteMRC($@"D:\EMD\9233\emd_9233_mask.mrc", true,Header);
           
            //sampledVol.Normalize();
            Header.PixelSize = Header.PixelSize * inVol.Dims.X / sampledVol.Dims.X;
            sampledVol.WriteMRC($@"D:\EMD\9233\emd_9233_1.5.mrc", true, Header);

        }

        static void Main(string[] args)
        {
            // The code provided will print ‘Hello World’ to the console.
            // Press Ctrl+F5 (or go to Debug > Start Without Debugging) to run your app.


            // Go to http://aka.ms/dotnet-get-started-console to continue learning how to build a console app!
            //downsampleToDim();
            //downsampler();
            //projectUniform();
            //projectUniform(@"D:\EMPIAR\10168\emd_4180_res7.mrc",  @"D:\EMPIAR\10168\shiny.star", @"D:\EMPIAR\10168\", @"D:\EMPIAR\10168\Projections_7_uniform")
            //preprocess_emd_9233();
            preprocess_emd_9233();
            projectUniform(@"D:\EMD\9233\emd_9233_Scaled_1.5.mrc", @"D:\EMPIAR\10168\shiny.star", @"D:\EMD\9233", @"D:\EMD\9233\Projections_1.5_uniform");

        }
    }
}
