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
using System.Globalization;

namespace Preprocessor
{
    class Preprocessor
    {
        static Random _rand = new Random();
        public static float NextGaussian()
        {
            float v1, v2, s;
            do
            {
                v1 = (2.0f * (float)_rand.NextDouble()) - 1.0f;
                v2 = 2.0f * (float)_rand.NextDouble() - 1.0f;
                s = v1 * v1 + v2 * v2;
            } while (s >= 1.0f || s == 0f);

            s = (float)Math.Sqrt((-2.0f * Math.Log(s)) / s);

            return v1 * s;
        }
        

        public static double Draw(double μ = 0.5, double σ = 0.5)
        {
            while (true)
            {
                // Get random values from interval [0,1]
                var x = _rand.NextDouble();
                var y = _rand.NextDouble();

                // Is the point (x,y) under the curve of the density function?
                if (y < f(x, μ, σ))
                    return x;
            }
        }

        // Normal (or gauss) distribution function
        public static double f(double x, double μ = 0.5, double σ = 0.5)
        {
            return 1d / Math.Sqrt(2 * σ * σ * Math.PI) * Math.Exp(-((x - μ) * (x - μ)) / (2 * σ * σ));
        }

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


        static void projectUniformMoved(string inputVolPath = @"D:\EMD\9233\emd_9233_Scaled_1.5_75k_bs64_it15_moving\", string starPath = @"D:\EMPIAR\10168\shiny.star", string outdir = @"D:\EMD\9233\emd_9233_Scaled_1.5_75k_bs64_it15_moving\", string projDir = @"D:\EMD\9233\emd_9233_Scaled_1.5_75k_bs64_it15_moving\Projections_1.5_uniform_256")
        {

            int numBatches = 10;
            int batchSize = 256;
            int numParticles = numBatches * batchSize;


            Image[] inVol = Helper.ArrayOfFunction(i=> Image.FromFile(inputVolPath + $"{i}.mrc"),numBatches);
            int2 projDim = new int2(inVol[0].Dims.X);




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
                Projector proj = new Projector(inVol[i], 3);
                Image im = proj.ProjectToRealspace(projDim, partAnglesRad);
               /* Image CTFs = new Image(im.Dims, true);
                GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                    CTFCoords.GetDevice(Intent.Read),
                    (uint)CTFCoords.ElementsSliceComplex,
                    CTFParams.Skip(((i) * batchSize)).Select(p => p.ToStruct()).ToArray(),
                    false,
                    (uint)im.Dims.Z);*/
                im.WriteMRC($@"{projDir}\{i}.mrc", true);
                //CTFs.WriteMRC($@"{projDir}\{i}_ctf.mrc", true);

                /* convolve once */
                /*Image imFFT = im.AsFFT();
                im.Dispose();
                imFFT.Multiply(CTFs);
                im = imFFT.AsIFFT();
                imFFT.Dispose();
                im.WriteMRC($@"{projDir}\{i}_convolved.mrc", true);
                */
                /* convolve with ctf^2 */
                /*imFFT = im.AsFFT();
                im.Dispose();
                imFFT.Multiply(CTFs);
                im = imFFT.AsIFFT();
                imFFT.Dispose();
                im.WriteMRC($@"{projDir}\{i}_convolved2.mrc", true);
                CTFs.Dispose();*/
                for (int j = 0; j < batchSize; j++)
                {
                    List<string> row = rows[n];
                    row[3] = $@"{j + 1}@{projDir}\{i}.mrc";
                    row[12] = partAnglesDeg[j].X.ToString();
                    row[13] = partAnglesDeg[j].Y.ToString();
                    row[14] = partAnglesDeg[j].Z.ToString();

                    starCleanOutFile.AddRow(new List<string>(row));
                    /*
                    row[3] = $@"{j + 1}@{projDir}\{i}_convolved.mrc";
                    starConvolvedOutFile.AddRow(new List<string>(row));

                    row[3] = $@"{j + 1}@{projDir}\{i}_convolved2.mrc";
                    starConvolved2OutFile.AddRow(new List<string>(row));*/
                    n++;
                }
                proj.Dispose();

            }
            starCleanOutFile.Save($@"{inputVolPath}projections_uniform_256.star");
            //starConvolvedOutFile.Save($@"{inputVolPath}projectionsConv_uniform.star");
            //starConvolved2OutFile.Save($@"{inputVolPath}projectionsConv2_uniform.star");

            /*
                string fileName = @"D:\EMD\9233\emd_9233_Scaled_1.5_40k_bs64_it3.pdb";
                string outFileName = @"D:\EMD\9233\emd_9233_Scaled_1.5_40k_bs64_it3_moved.pdb";
                IEnumerable<string> lines = File.ReadLines(fileName);
                List<float3> atomPositions = new List<float3>();
                List<float3> atomPositionsMoved = new List<float3>();
                List<double> intensities = new List<double>();
                List<string> remarkLines = new List<string>();
                float zMin = 100000, zMax = -1;
                double sigma = 0.0;
                double minInt = 0.0;
                double scaleF = 0.0;
            foreach(var line in lines)
            {
                if (line.StartsWith("REMARK"))
                {
                    remarkLines.Add(line);
                    if(line.StartsWith("REMARK fixedGaussian"))
                    {
                        sigma = double.Parse(line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[2]);
                    }
                    if (line.StartsWith("REMARK Scaled ints"))
                    {
                        scaleF = double.Parse(line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[3]);
                    }
                    if (line.StartsWith("REMARK min int"))
                    {
                        minInt = double.Parse(line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)[3]);
                    }
                    continue; //continue 
                }
                else if (line.StartsWith("ATOM"))
                {
                    float[] res = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Skip(5).Take(4).Select(s => float.Parse(s, CultureInfo.InvariantCulture.NumberFormat)).ToArray();
                    float3 pos = new float3(res[0], res[1], res[2]);
                    double intensity = res[3];
                    atomPositions.Add(pos);
                    intensities.Add(intensity);

                    zMin = Math.Min(pos.Z, zMin);
                    zMax = Math.Max(pos.Z, zMax);
                }
            }

            double a = Math.Pow(zMax - zMin, 2);

            for(int i=0; i< atomPositions.Count(); i++)
            {
                atomPositionsMoved.Add(new float3((float)(atomPositions[i].X + 0*Math.Pow(atomPositions[i].Z - zMin, 2) / a), atomPositions[i].Y, atomPositions[i].Z));
            }
            Image im = Image.FromFile(@"D:\EMD\9233\emd_9233_Scaled_1.5.mrc");
            im.Multiply(0.0f);
            float[][] data = im.GetHost(Intent.Write);

            for(int i=0; i<intensities.Count();i++)
            {
                float3 pos = atomPositions[i];
                int startX = (int)Math.Max(0, Math.Ceiling(pos.X - 4 * sigma));
                int endX = (int)Math.Min(im.Dims.X, Math.Floor(pos.X + 4 * sigma));

                int startY = (int)Math.Max(0,Math.Ceiling(pos.Y - 4 * sigma));
                int endY = (int)Math.Min(im.Dims.Y, Math.Floor(pos.Y + 4 * sigma));

                int startZ = (int)Math.Max(0,Math.Ceiling(pos.Z - 4 * sigma));
                int endZ = (int)Math.Min(im.Dims.Z, Math.Floor(pos.Z + 4 * sigma));
                for (int x=startX; x<=endX; x++)
                {
                    double xx = Math.Pow(x - pos.X, 2);
                    for (int y = startY; y <= endY; y++)
                    {
                        double yy = Math.Pow(y - pos.Y,2);
                        for (int z = startZ; z <= endZ; z++)
                        {
                            double zz = Math.Pow(z - pos.Z,2);
                            data[z][y * im.Dims.X + x] += (float)(intensities[i] * Math.Exp(-(xx + yy + zz) / (Math.Pow(sigma, 2))));
                        }
                    }

                }



            }*/
            //im.WriteMRC(@"D:\EMD\9233\emd_9233_Scaled_1.5_moved.mrc", true);
            /*
            using (System.IO.StreamWriter file =
            new System.IO.StreamWriter(outFileName))
            {
                foreach (var line in remarkLines)
                {
                    file.WriteLine(line);
                }
                for (int i = 0; i < intensities.Count(); i++)
                {
                    file.WriteLine($"ATOM      {i + 1} DENS DENS       {i + 1}      {atomPositions[i].X:8.3}   {atomPositions[i].Y:8.3}   {atomPositions[i].Z:8.3} {intensities[i]:.8}     1      DENS");
                }

            }*/



        }

        static void projectUniform(string inputVolPath = @"D:\EMPIAR\10168\emd_4180_res7.mrc", string starPath = @"D:\EMPIAR\10168\shiny.star", string outdir = @"D:\EMPIAR\10168\", string projDir = @"D:\EMPIAR\10168\Projections_7_uniform", int numParticles = 500)
        {
            int batchSize = 1024;
            int numBatches = (int)(Math.Ceiling((double)numParticles/batchSize));

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

            CTF[] CTFParams = starInFile.GetRelionCTF();
            Image CTFCoords = CTF.GetCTFCoords(projDim, projDim, 3.0f);

            int numAngles = numParticles;
            int numAnglesX = (int)Math.Ceiling(Math.Pow(numAngles, 1.0d / 2.0d));
            int numAnglesY = (int)Math.Max(1, Math.Floor(Math.Pow(numAngles, 1.0d / 2.0d)));
            numAngles = numAnglesX * numAnglesY;
            numParticles = numAngles;
            float zz = 0;
            float3[] anglesRad = new float3[numAngles];
            {
                int i = 0;
                for (int x = 0; x < numAnglesX; x++)
                {
                    float xx = (float)(2 * Math.PI) / (numAnglesX - 1) * (x);
                    for (int y = 0; y < numAnglesY; y++)
                    {
                        float yy = (float)(2 * Math.PI) / (numAnglesY - 1) * (y);
                        anglesRad[i] = new float3(xx, yy, zz);
                        i++;

                    }
                }
            }

            float3[] anglesDeg = Helper.ArrayOfFunction(i => anglesRad[i] * Helper.ToDeg, anglesRad.Count());

            List<List<string>> rows = starInFile.GetAllRows();
            int n = 0;
            Image CTFSum = new Image(new int3(projDim), true);
            Image[] outProjections = new Image[numBatches];
            Image[] outConvolved = new Image[numBatches];
            Image[] outCTFs = new Image[numBatches];
            for (int i = 0; i < numBatches; i++)
            {
                int batchElements = Math.Min(batchSize, numAngles - batchSize);
                float3[] partAnglesRad = anglesRad.Skip(((i) * batchSize)).Take(batchElements).ToArray();

                float3[] partAnglesDeg = Helper.ArrayOfFunction(k => partAnglesRad[k] * Helper.ToDeg, batchElements);
                Image im = proj.ProjectToRealspace(projDim, partAnglesRad);
                Image CTFs = new Image(im.Dims, true);
                GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                    CTFCoords.GetDevice(Intent.Read),
                    (uint)CTFCoords.ElementsSliceComplex,
                    CTFParams.Skip(((i) * batchSize)).Select(p => p.ToStruct()).ToArray(),
                    false,
                    (uint)im.Dims.Z);
                outProjections[i] = im;
                outCTFs[i] = CTFs;
                /* convolve once */
                Image imFFT = im.AsFFT();
                imFFT.Multiply(CTFs);
                im = imFFT.AsIFFT();
                imFFT.Dispose();
                outConvolved[i] = im;

                for (int j = 0; j < batchElements; j++)
                {
                    List<string> row = rows[n];
                    row[3] = $@"{n + 1}@{projDir}\projections_uniform.mrc";
                    row[12] = partAnglesDeg[j].X.ToString();
                    row[13] = partAnglesDeg[j].Y.ToString();
                    row[14] = partAnglesDeg[j].Z.ToString();

                    starCleanOutFile.AddRow(new List<string>(row));

                    row[3] = $@"{n + 1}@{projDir}\projections_uniform_convolved.mrc";
                    starConvolvedOutFile.AddRow(new List<string>(row));

                    n++;
                }
                
            }
            Image.Stack(outProjections).WriteMRC($@"{projDir}\projections_uniform.mrc");
            Image.Stack(outConvolved).WriteMRC($@"{projDir}\projections_uniform_convolved.mrc");
            Image.Stack(outCTFs).WriteMRC($@"{projDir}\projections_uniform_ctf.mrc");

            starCleanOutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projections_uniform.star");
            starConvolvedOutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projections_uniform_convolved.star");
        }


        static void projectTomo(string inputVolPath = @"D:\EMPIAR\10168\emd_4180_res7.mrc", string starPath = @"D:\EMPIAR\10168\shiny.star", string outdir = @"D:\EMPIAR\10168\", string projDir = @"D:\EMPIAR\10168\Projections_7_uniform", int order=4)
        {
            int batchSize = 1024;


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

            CTF[] CTFParams = starInFile.GetRelionCTF();
            Image CTFCoords = CTF.GetCTFCoords(projDim, projDim, 3.0f);

            float3[] anglesRad = Helper.GetHealpixRotTilt(order).Select(v => new float3(v.X, v.Y, 0) * Helper.ToRad).ToArray();
            int numAngles = anglesRad.Length;
            int numParticles = numAngles;
            int numBatches = (int)(Math.Ceiling((double)numParticles / batchSize));

            float3[] anglesDeg = Helper.ArrayOfFunction(i => anglesRad[i] * Helper.ToDeg, anglesRad.Count());

            List<List<string>> rows = starInFile.GetAllRows();
            int n = 0;
            Image CTFSum = new Image(new int3(projDim), true);
            Image[] outProjections = new Image[numBatches];
            Image[] outConvolved = new Image[numBatches];
            Image[] outCTFs = new Image[numBatches];
            for (int i = 0; i < numBatches; i++)
            {
                int batchElements = Math.Min(batchSize, numAngles - batchSize);
                float3[] partAnglesRad = anglesRad.Skip(((i) * batchSize)).Take(batchElements).ToArray();

                float3[] partAnglesDeg = Helper.ArrayOfFunction(k => partAnglesRad[k] * Helper.ToDeg, batchElements);
                Image im = proj.ProjectToRealspace(projDim, partAnglesRad);
                Image CTFs = new Image(im.Dims, true);
                GPU.CreateCTF(CTFs.GetDevice(Intent.Write),
                    CTFCoords.GetDevice(Intent.Read),
                    (uint)CTFCoords.ElementsSliceComplex,
                    CTFParams.Skip(((i) * batchSize)).Select(p => p.ToStruct()).ToArray(),
                    false,
                    (uint)im.Dims.Z);
                outProjections[i] = im;
                outCTFs[i] = CTFs;
                /* convolve once */
                Image imFFT = im.AsFFT();
                imFFT.ShiftSlices(Helper.ArrayOfFunction(idx => new float3(im.Dims.X / 2, im.Dims.Y / 2, 0), batchElements));
                imFFT.Multiply(CTFs);
                imFFT.ShiftSlices(Helper.ArrayOfFunction(idx => new float3(-im.Dims.X / 2, -im.Dims.Y / 2, 0), batchElements));
                im = imFFT.AsIFFT();
                imFFT.Dispose();
                outConvolved[i] = im;

                for (int j = 0; j < batchElements; j++)
                {
                    List<string> row = rows[n];
                    row[3] = $@"{n + 1}@{projDir}\projections_tomo.mrc";
                    row[12] = partAnglesDeg[j].X.ToString();
                    row[13] = partAnglesDeg[j].Y.ToString();
                    row[14] = partAnglesDeg[j].Z.ToString();

                    starCleanOutFile.AddRow(new List<string>(row));

                    row[3] = $@"{n + 1}@{projDir}\projections_tomo_convolvedShift.mrc";
                    starConvolvedOutFile.AddRow(new List<string>(row));

                    n++;
                }

            }
            Image.Stack(outProjections).WriteMRC($@"{projDir}\projections_tomo.mrc");
            Image.Stack(outConvolved).WriteMRC($@"{projDir}\projections_tomo_convolvedShift.mrc");
            Image.Stack(outCTFs).WriteMRC($@"{projDir}\projections_uniform_ctf.mrc");

            starCleanOutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projections_tomo.star");
            starConvolvedOutFile.Save($@"{inputVolPath.Replace(".mrc", "")}.projections_tomo_convolvedShift.star");
        }


        static void projectorByStar(string inputVolPath = @"D:\EMPIAR\10168\emd_4180_res7.mrc", string outdir = @"D:\EMPIAR\10168\",  string projDir = @"D:\EMPIAR\10168\Projections_7", string starPath = @"D:\EMPIAR\10299\data\Particles\shiny.star")
        {
            
            Image inVol = Image.FromFile(inputVolPath);
            Projector proj = new Projector(inVol, 3);
 
            if (!Directory.Exists(projDir))
            {
                Directory.CreateDirectory(projDir);
            }
            int2 projDim = new int2(inVol.Dims.X);
            int batchSize = 1024;
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
            float3 newPix = new float3(2.0f);
            Image sampledVol = ImageProcessor.Downsample(inVol, newPix.X / olPix.X);
            Image scaledVol = inVol.AsScaled(sampledVol.Dims);
            Image scaledMask = scaledVol.GetCopy();
            scaledMask.Binarize(0.01f);
            scaledMask = scaledMask.AsConvolvedGaussian(2.0f);
            scaledMask.Binarize(0.2f);
            scaledMask = scaledMask.AsConvolvedGaussian(2.0f);
            scaledMask.Binarize(0.2f);
            scaledMask = scaledMask.AsConvolvedGaussian(10.0f);
            scaledMask.Binarize(0.2f);
            scaledMask.WriteMRC($@"D:\EMD\9233\emd_9233_Scaled_2.0_mask.mrc");
            scaledVol.Multiply(scaledMask);
            scaledVol.WriteMRC($@"D:\EMD\9233\emd_9233_Scaled_2.0.mrc");
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
            /*
            Header.PixelSize = Header.PixelSize * inVol.Dims.X / sampledVol.Dims.X;
            sampledVol.WriteMRC($@"D:\EMD\9233\emd_9233_1.5.mrc", true, Header);
            */
        }


        static void distort(String projectionsFileName)
        {
            Image projections = Image.FromFile(projectionsFileName);

            float[][] noiseData = Helper.ArrayOfFunction(i => new float[projections.Dims.ElementsSlice()], projections.Dims.Z);

            float[][] projectionsData = projections.GetHost(Intent.Read);

            float min = 100000;
            float max = -1000000;
            double avgSqrd = 0.0;

            for (int z = 0; z < projections.Dims.Z; z++)
            {
                for (int y = 0; y < projections.Dims.Y; y++)
                {
                    for (int x = 0; x < projections.Dims.X; x++)
                    {
                        max = Math.Max(max, projectionsData[z][y * projections.Dims.X + x]);
                        min = Math.Min(min, projectionsData[z][y * projections.Dims.X + x]);
                        avgSqrd += Math.Pow(projectionsData[z][y * projections.Dims.X + x], 2);
                    }
                }
            }
            avgSqrd /= projections.ElementsReal;
            for (int i = 100; i <= 1000; i+=100)
            {
                float SNR = 1.0f / i;
                Image projDistorted = projections.GetCopy();
                Random rand = new Random(123);
                for (int z = 0; z < projections.Dims.Z; z++)
                {
                    for (int y = 0; y < projections.Dims.Y; y++)
                    {
                        for (int x = 0; x < projections.Dims.X; x++)
                        {
                            if (min < 0)
                            {
                                noiseData[z][y * projections.Dims.X + x] = (float)Draw(i * avgSqrd, (max - min) / 8);
                            }
                            else
                                noiseData[z][y * projections.Dims.X + x] = (float)Draw(i * avgSqrd, (max - min) / 8);
                        }
                    }
                }
                Image noise = new Image(noiseData, projections.Dims);
                noise.WriteMRC(projectionsFileName.Replace(".mrc", $".noise_{i}.mrc"), true);
                projDistorted.Add(noise);
                projDistorted.WriteMRC(projectionsFileName.Replace(".mrc", $".distorted_{i}.mrc"),true);
            }

        }

        static void Main(string[] args)
        {
            // The code provided will print ‘Hello World’ to the console.
            // Press Ctrl+F5 (or go to Debug > Start Without Debugging) to run your app.
            /*
                        Star starInFile = new Star(@"D:\EMD\9233\emd_9233_Scaled_2.0.projections_uniform.star");
                        Star starOutFile = new Star(starInFile.GetColumnNames());

                        List<List<string>> rows = starInFile.GetAllRows();
                        for (int j = 0; j < rows.Count; j++)
                        {
                            List<string> row = rows[j];https://intranet.mpibpc.mpg.de/
                            row[3] = $@"{j + 1}@D:\EMD\9233\Projections_2.0_uniform\combined.mrc";
                            starOutFile.AddRow(row);
                        }
                        starOutFile.Save(@"D:\EMD\9233\emd_9233_Scaled_2.0.projections_uniform_combined.star");*/
            // Go to http://aka.ms/dotnet-get-started-console to continue learning how to build a console app!
            //downsampleToDim();
            //downsampler();
            //preprocess_emd_9233();
            //projectUniform();
            //projectUniform(@"D:\EMPIAR\10168\emd_4180_res7.mrc",  @"D:\EMPIAR\10168\shiny.star", @"D:\EMPIAR\10168\", @"D:\EMPIAR\10168\Projections_7_uniform")
            //preprocess_emd_9233();
            //preprocess_emd_9233();
            //projectUniform(@"D:\EMD\9233\emd_9233_Scaled_2.0.mrc", @"D:\EMPIAR\10168\shiny.star", @"D:\EMD\9233", @"D:\EMD\9233\Projections_2.0_uniform", 3*1024);
            //projectTomo(@"D:\EMD\9233\emd_9233_Scaled_2.0.mrc", @"D:\EMPIAR\10168\shiny.star", @"D:\EMD\9233", @"D:\EMD\9233\Projections_2.0_tomo", 4);
            //distort($@"D:\EMD\9233\Projections_2.0_tomo\projections_tomo.mrc");
            //projectUniform(@"D:\EMD\9233\emd_9233_Scaled_1.2.mrc", @"D:\EMPIAR\10168\shiny.star", @"D:\EMD\9233", @"D:\EMD\9233\Projections_1.2_uniform");
            //projectUniformMoved();
            //Image Ref = Image.FromFile(@"D:\EMD\9233\emd_9233_Scaled_2.0.mrc");
            //Image mask = new Image(Ref.Dims);
            //mask.TransformValues((x, y, z, v) => {
            //   if (Math.Abs(z-62) < 53 && Math.Sqrt(Math.Pow(x-67,2) + Math.Pow(y-67, 2)) < 45)
            //       return 1.0f;
            //    return 0.0f;
            //});
            //mask.WriteMRC(@"D:\EMD\9233\emd_9233_Scaled_2.0_largeMask.mrc", true);
            projectorByStar(@"D:\EMD\9233\TomoReconstructions\2.0_convolvedFromAtoms.1024_it1_oversampled4.mrc", @"D:\EMD\9233\TomoReconstructions", @"D:\EMD\9233\TomoReconstructions\2.0_convolvedFromAtoms.1024_it1_oversampled4_projections", @"D:\EMD\9233\emd_9233_Scaled_2.0.projections_tomo_convolved-fromAtoms.1024.star");
        }
    }
}
