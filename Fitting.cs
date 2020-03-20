using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

using Warp;
using Warp.Tools;
using ProjClassifier.Tools;
using FlexibleRefinement.Util;
using System.Globalization;
using System.Diagnostics;
using CommandLine;

namespace FlexibleRefinement
{
    class Fitting
    {

        

        static void stickToArc()
        {
            String trial = "Stick_to_Arc";
            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            /* Load Images */
            //simulate();
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image arcIm = Image.FromFile("ArcVolume_Created.mrc");
            GPU.Normalize(arcIm.GetDevice(Intent.Read),
                         arcIm.GetDevice(Intent.Write),
                         (uint)arcIm.ElementsReal,
                         (uint)1);
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            Image arcMask = arcIm.GetCopy();
            arcMask.Binarize((float)(1.0f / Math.E));
            arcMask.WriteMRC($@"{trial}\arcMask.mrc");
            /* Create Graphs */
            AtomGraph startGraph = new AtomGraph(stickIm, stickMask, 1000);
            AtomGraph targetGraph = new AtomGraph(arcIm, arcMask, 1000);

            /* iterate Movements */


            targetGraph.Repr().WriteMRC($@"{trial}\{trial}_Target.mrc");
            startGraph.Repr().WriteMRC($@"{trial}\{trial}_Start.mrc");

            for (int j = 0; j < 10; j++)
            {
                int i = 0;
                arcMask = Image.FromFile($"Arc.Pi.6.{j}Mask_Created.mrc");
                startGraph.setEMIntensities(arcMask.AsConvolvedGaussian(5));
                for (; i < 5; i++)
                {
                    startGraph.moveAtoms(5.0f, 2.0f);
                    startGraph.Repr().WriteMRC($@"{trial}\{trial}_{j}_im_it{i + 1}.mrc");
                }
                startGraph.setEMIntensities(arcMask.AsConvolvedGaussian(1));
                for (; i < 10; i++)
                {
                    startGraph.moveAtoms(10.0f, 4.0f);
                    startGraph.Repr().WriteMRC($@"{trial}\{trial}_{j}_im_it{i + 1}.mrc");
                }
                startGraph.save($@"{trial}\{trial}_{j}.graph");
            }
        }

        private static void AddText(FileStream fs, string value)
        {
            byte[] info = new UTF8Encoding(true).GetBytes(value);
            fs.Write(info, 0, info.Length);
        }

        private static void EvalStickToArc()
        {

            String trial = "Stick_to_Arc";
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph initial = new AtomGraph($@"{trial}\Stick_to_Arc_initial.graph", stickIm);
            AtomGraph[] loaded = Helper.ArrayOfFunction(i => new AtomGraph($@"{trial}\Stick_to_Arc_{i}.graph", stickIm), 10);

            float3[][] displacements = Helper.ArrayOfFunction(i => Helper.ArrayOfFunction(j => new float3(0.0f), loaded[0].Atoms.Count), 10);
            List<float>[] atomDistances = Helper.ArrayOfFunction(i => new List<float>(), 10);
            using (FileStream fs = File.Create($@"{trial}\{trial}_distanceList_initial.txt"))
            {
                for (int j = 0; j < initial.Atoms.Count; j++)
                {
                    foreach (var atom in initial.Atoms[j].Neighbours)
                    {
                        AddText(fs, $"{(atom.Pos - initial.Atoms[j].Pos).Length()}\n".Replace(",", "."));
                    }
                }
            }
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < initial.Atoms.Count; j++)
                {
                    displacements[i][j] = loaded[i].Atoms[j].Pos - initial.Atoms[j].Pos;
                    foreach (var atom in loaded[i].Atoms[j].Neighbours)
                    {
                        atomDistances[i].Add((atom.Pos - loaded[i].Atoms[j].Pos).Length());
                    }
                }
            }
            for (int i = 0; i < 10; i++)
            {
                using (FileStream fs = File.Create($@"{trial}\{trial}_distanceList_{i}.txt"))
                {
                    foreach (var d in atomDistances[i])
                    {
                        AddText(fs, $"{d}\n".Replace(',', '.'));
                    }

                }

                using (FileStream fs = File.Create($@"{trial}\{trial}_displacementList_{i}.txt"))
                {
                    foreach (var d in displacements[i])
                    {
                        AddText(fs, $"{d.X} {d.Y} {d.Z}\n".Replace(',', '.'));
                    }
                }

            }

        }

        private static void rotateGraph(String initialFile, String finalFile, String displacementFile, float c)
        {
            String trial = "RotateStick"; if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            //Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            //Image stickMask = stickIm.GetCopy();
            //stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph initial = new AtomGraph(initialFile, null);
            var atomList = initial.Atoms;
            float3 com = new float3(0);
            foreach (var atom in atomList)
            {
                com = com + atom.Pos;
            }

            com = com / atomList.Count;
            using (FileStream fs = File.Create(displacementFile))
            {

                foreach (var atom in atomList)
                {
                    float3 before = atom.Pos;
                    atom.Pos = com + Helpers.rotate_euler(atom.Pos - com, new float3((float)(Math.PI / c), 0, 0));
                    float3 after = atom.Pos;
                    float3 gtDisplacement = after - before;
                    AddText(fs, $"{gtDisplacement.X} {gtDisplacement.Y} {gtDisplacement.Z}\n".Replace(',', '.'));
                }
            }
            initial.save(finalFile);
            //initial.Repr().WriteMRC(finalFile.Replace(".graph", ".mrc"));
        }

        private static void createRotated(int c)
        {
            String trial = "RotateStick"; if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = stickIm.GetCopy();
            stickMask.Binarize((float)(1.0f / Math.E));
            AtomGraph initial = new AtomGraph($@"{trial}\Stick_Initial.graph", stickIm);
            var atomList = initial.Atoms;
            float3 com = new float3(0);
            foreach (var atom in atomList)
            {
                com = com + atom.Pos;
            }

            com = com / atomList.Count;
            using (FileStream fs = File.Create($@"{trial}\{trial}_gtDisplacementList_PI_{c}.txt"))
            {

                foreach (var atom in atomList)
                {
                    float3 before = atom.Pos;
                    atom.Pos = com + Helpers.rotate_euler(atom.Pos - com, new float3((float)(Math.PI / c), 0, 0));
                    float3 after = atom.Pos;
                    float3 gtDisplacement = after - before;
                    AddText(fs, $"{gtDisplacement.X} {gtDisplacement.Y} {gtDisplacement.Z}\n".Replace(',', '.'));
                }
            }
            initial.save($@"{trial}\Rotate_PI_{c}_gt.graph");
            //initial.Repr().WriteMRC($@"{trial}\Rotate_PI_{c}_gt.mrc");
        }



        


        public static void doRotationExp(String outdir, float c, List<float> corrScales, List<float> distScales, List<bool> normalizes)
        {
            float pixSize = 1.0f;
            if (!Directory.Exists(outdir))
            {
                Directory.CreateDirectory(outdir);
            }
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = Image.FromFile("StickMask_Created.mrc");

            Image stickRotIm = Image.FromFile($"StickVolume_Rotated_{c}_Created.mrc");
            Image stickRotMask = Image.FromFile($"StickMask_Rotated_{c}_Created.mrc");

            float[] sampleRates = new float[3] { 4, 2, 1 };
            int targetCount = 10000;
            int[] sampledCounts = Helper.ArrayOfFunction(k => (int)(targetCount / Math.Pow(sampleRates[k], 3)), 3);
            Image[] StartIms = Helper.ArrayOfFunction(k => sampleRates[k] == 1 ? stickIm.GetCopy() : ImageProcessor.Downsample(stickIm, sampleRates[k]), 3);
            Image[] StartMasks = Helper.ArrayOfFunction(k =>
            {
                Image mask = StartIms[k].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);

            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(k => new AtomGraph(StartIms[k], StartMasks[k], sampledCounts[k]), 3);


            Image[] TarIms = Helper.ArrayOfFunction(k => sampleRates[k] == 1 ? stickRotIm.GetCopy() : ImageProcessor.Downsample(stickRotIm, sampleRates[k]), 3);
            Image[] TarMasks = Helper.ArrayOfFunction(k =>
            {
                Image mask = TarIms[k].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);
            for (int k = 0; k < 3; k++)
            {
                StartIms[k].WriteMRC($@"{outdir}\{sampleRates[k]}_StartIm.mrc");
                TarIms[k].WriteMRC($@"{outdir}\{sampleRates[k]}_{c}_TarIm.mrc");

                StartMasks[k].WriteMRC($@"{outdir}\{sampleRates[k]}_StartMask.mrc");
                TarMasks[k].WriteMRC($@"{outdir}\{sampleRates[k]}_{c}_TarMask.mrc");

                StartGraphs[k].save($@"{outdir}\{sampleRates[k]}_StartGraph.graph");
                //StartGraphs[k].Repr().WriteMRC($@"{outdir}\{sampleRates[k]}_StartGraph.mrc");

                rotateGraph($@"{outdir}\{sampleRates[k]}_StartGraph.graph", $@"{outdir}\{sampleRates[k]}_{c}_TarGraph.graph", $@"{outdir}\{sampleRates[k]}_{c}_GtDisplacements", c);
                AtomGraph targetGraph = new AtomGraph($@"{outdir}\{sampleRates[k]}_{c}_TarGraph.graph", TarIms[k]);
                //targetGraph.Repr().WriteMRC($@"{outdir}\{sampleRates[k]}_{c}_TarGraph.mrc");
            }



            #region FirstStep
            if (!Directory.Exists($@"{outdir}\StepOne"))
            {
                Directory.CreateDirectory($@"{outdir}\StepOne");
            }
            TarIms[0].AsConvolvedGaussian(1).WriteMRC($@"{outdir}\StepOne\{sampleRates[0]}_{c}_TarIm_Convolved.mrc");


            DateTime begin = DateTime.UtcNow;

            int i = 0;
            AtomGraph localStartGraph = new AtomGraph($@"{outdir}\{sampleRates[0]}_StartGraph.graph", TarIms[0].AsConvolvedGaussian(1));
            for (; i < 10; i++)
            {
                localStartGraph.moveAtoms(corrScales[0], distScales[0], normalizes[0]);
                localStartGraph.save($@"{outdir}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScales[0]:#.#}_{distScales[0]:#.#}_{normalizes[0]}_it{i + 1}.graph");
            }
            localStartGraph.save($@"{outdir}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScales[0]:#.#}_{distScales[0]:#.#}_{normalizes[0]}_final.graph");

            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion


            #region SecondStep
            String fromFirstGraphFileStart = $@"{outdir}\{sampleRates[0]}_StartGraph.graph";
            AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
            String fromFirstGraphFileFinal = $@"{outdir}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScales[0]:#.#}_{distScales[0]:#.#}_{normalizes[0]}_final.graph";
            AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

            List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
            for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{outdir}\StepTwo"))
            {
                Directory.CreateDirectory($@"{outdir}\StepTwo");
            }
            begin = DateTime.UtcNow;

            i = 0;
            localStartGraph = new AtomGraph($@"{outdir}\{sampleRates[1]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
            localStartGraph.setPositions(fromFirstGraphStart, displacements);
            for (; i < 10; i++)
            {
                localStartGraph.moveAtoms(corrScales[1], distScales[1], normalizes[1]);
                localStartGraph.save($@"{outdir}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScales[1]:#.#}_{distScales[1]:#.#}_{normalizes[1]}_it{i + 1}.graph");
            }
            localStartGraph.save($@"{outdir}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScales[1]:#.#}_{distScales[1]:#.#}_{normalizes[1]}_final.graph");

            end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion



            #region ThirdStep
            String fromSecondGraphFileStart = $@"{outdir}\{sampleRates[1]}_StartGraph.graph";
            AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
            String fromSecondGraphFileFinal = $@"{outdir}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScales[1]:#.#}_{distScales[1]:#.#}_{normalizes[1]}_final.graph";
            AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

            displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
            for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{outdir}\StepThree"))
            {
                Directory.CreateDirectory($@"{outdir}\StepThree");
            }
            begin = DateTime.UtcNow;

            i = 0;
            localStartGraph = new AtomGraph($@"{outdir}\{sampleRates[2]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
            localStartGraph.setPositions(fromSecondGraphStart, displacements);
            for (; i < 10; i++)
            {
                localStartGraph.moveAtoms(corrScales[2], distScales[2], normalizes[2]);
                localStartGraph.save($@"{outdir}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScales[2]:#.#}_{distScales[2]:#.#}_{normalizes[2]}_it{i + 1}.graph");
            }
            localStartGraph.save($@"{outdir}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScales[2]:#.#}_{distScales[2]:#.#}_{normalizes[2]}_final.graph");

            end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion*/
        }

        public static void GridSearchParams(int c=10)
        {
            String trial = "GridSearchParams_No_NeighborUpdate_c10000_it100";
            float pixSize = 1.0f;

            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            Image stickMask = Image.FromFile("StickMask_Created.mrc");


            Image stickRotIm = Image.FromFile($"StickVolume_Rotated_{c}_Created.mrc");
            Image stickRotMask = Image.FromFile($"StickMask_Rotated_{c}_Created.mrc");

            float[] sampleRates = new float[3] { 4, 2, 1 };
            int targetCount = 10000;
            int[] sampledCounts = Helper.ArrayOfFunction(i => (int)(targetCount / Math.Pow(sampleRates[i], 3)), 3);
            Image[] StartIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? stickIm.GetCopy() : ImageProcessor.Downsample(stickIm, sampleRates[i]), 3);
            Image[] StartMasks = Helper.ArrayOfFunction(i => {
                Image mask = StartIms[i].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);

            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(i => new AtomGraph(StartIms[i], StartMasks[i], sampledCounts[i]), 3);


            Image[] TarIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? stickRotIm.GetCopy() : ImageProcessor.Downsample(stickRotIm, sampleRates[i]), 3);
            Image[] TarMasks = Helper.ArrayOfFunction(i => {
                Image mask = TarIms[i].GetCopy();
                mask.Binarize(0.25f);
                return mask;
            }, 3);
            for (int i = 0; i < 3; i++)
            {
                StartIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartIm.mrc");
                TarIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarIm.mrc");

                StartMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartMask.mrc");
                TarMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarMask.mrc");

                StartGraphs[i].save($@"{trial}\{sampleRates[i]}_StartGraph.graph");
                //StartGraphs[i].Repr().WriteMRC($@"{trial}\{sampleRates[i]}_StartGraph.mrc");

                rotateGraph($@"{trial}\{sampleRates[i]}_StartGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_GtDisplacements", c);
                AtomGraph targetGraph = new AtomGraph($@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", TarIms[i]);
               // targetGraph.Repr().WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarGraph.mrc");
            }

            float[] corrScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            bool[] normalizings = new bool[2] { true, false };
            /*
            #region FirstStep
            if (!Directory.Exists($@"{trial}\StepOne"))
            {
                Directory.CreateDirectory($@"{trial}\StepOne");
            }
            TarIms[0].AsConvolvedGaussian(1).WriteMRC($@"{trial}\StepOne\{sampleRates[0]}_{c}_TarIm_Convolved.mrc");

            
            DateTime begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) => {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[0]}_StartGraph.graph", TarIms[0].AsConvolvedGaussian(1));
                        for (;i < 100; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }

            }, null);
            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion
            */
            /*
           #region SecondStep
           String fromFirstGraphFileStart = $@"{trial}\{sampleRates[0]}_StartGraph.graph";
           AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
           String fromFirstGraphFileFinal = $@"{trial}\StepOne\4_Rotate_PI_{c}_1_10_False_final.graph";
           AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

           List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
           for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
           {
               displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
           }


           
           if (!Directory.Exists($@"{trial}\StepTwo"))
           {
               Directory.CreateDirectory($@"{trial}\StepTwo");
           }
           DateTime begin = DateTime.UtcNow;
           Helper.ForCPU(0, 20, 20, null, (k, id, ts) => {
               float corrScale = corrScales[k];

               foreach (var distScale in distScales)
               {
                   foreach (var normalizing in normalizings)
                   {
                       int i = 0;
                       AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[1]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
                       localStartGraph.setPositions(fromFirstGraphStart, displacements);
                       for (; i < 100; i++)
                       {
                           localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                           localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                       }
                       localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                   }

               }

           }, null);
           DateTime end = DateTime.UtcNow;
           System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
           #endregion
           */

            
            #region ThirdStep
            String fromSecondGraphFileStart = $@"{trial}\{sampleRates[1]}_StartGraph.graph";
           AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
           String fromSecondGraphFileFinal = $@"{trial}\StepTwo\2_Rotate_PI_{c}_7_9_False_final.graph";
           AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

           List<float3> displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
           for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
           {
               displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
           }



           if (!Directory.Exists($@"{trial}\StepThree"))
           {
               Directory.CreateDirectory($@"{trial}\StepThree");
           }
           DateTime begin = DateTime.UtcNow;
           Helper.ForCPU(0, 20, 20, null, (k, id, ts) => {
               float corrScale = corrScales[k];

               foreach (var distScale in distScales)
               {
                   foreach (var normalizing in normalizings)
                   {
                       int i = 0;
                       AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[2]}_StartGraph.graph", TarMasks[1].AsConvolvedGaussian(1));
                       localStartGraph.setPositions(fromSecondGraphStart, displacements);
                       for (; i < 50; i++)
                       {
                           localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                           localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                       }
                       localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                   }

               }

           }, null);
           DateTime end = DateTime.UtcNow;
           System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
           #endregion
          
           
            return;
        }

        public static void GridSearchParamsRot(String startImFile, String startMaskFile, String tarImFile, String tarMaskFile, String trial, int c = 10, int targetCount = 2000)
        {
            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            int steps = 3;
            float[] sampleRates = new float[3] { 4, 2, 1 };
            int[] numIterations = new int[3] { 50, 50, 50 };
            int[] sampledCounts = Helper.ArrayOfFunction(i => (int)(targetCount / sampleRates[i]), 3);
            float[] corrScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            bool[] normalizings = new bool[2] { true, false };

            float[] minCorrScales = new float[steps];
            float[] minDistScales = new float[steps];
            bool[] minNormalizings = new bool[steps];


            Image startIm = Image.FromFile(startImFile);
            Image startMask = Image.FromFile(startMaskFile);
            Image tarIm = Image.FromFile(tarImFile);
            Image tarMask = Image.FromFile(tarMaskFile);



            Image[] StartIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? startIm.GetCopy() : ImageProcessor.Downsample(startIm, sampleRates[i]), steps);
            Image[] StartMasks = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? startMask.GetCopy() : ImageProcessor.Downsample(startMask, sampleRates[i]), steps);
            for (int k = 0; k < steps; k++)
            {
                StartMasks[k].Binarize(1.0f);
            }
            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(i => new AtomGraph(StartIms[i], StartMasks[i], sampledCounts[i]), steps);


            Image[] TarIms = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? tarIm.GetCopy() : ImageProcessor.Downsample(tarIm, sampleRates[i]), steps);
            Image[] TarMasks = Helper.ArrayOfFunction(i => sampleRates[i] == 1 ? tarMask.GetCopy() : ImageProcessor.Downsample(tarMask, sampleRates[i]), steps);
            for (int k = 0; k < steps; k++)
            {
                TarMasks[k].Binarize(1.0f);
            }


            //AtomGraph[] TargetGraphs = Helper.ArrayOfFunction(i => new AtomGraph(TarIms[i], TarMasks[i], sampledCounts[i]), steps);

            for (int i = 0; i < steps; i++)
            {
                StartIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartIm.mrc");
                TarIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarIm.mrc");

                StartMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartMask.mrc");
                TarMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_{c}_TarMask.mrc");

                StartGraphs[i].save($@"{trial}\{sampleRates[i]}_StartGraph.graph");

                rotateGraph($@"{trial}\{sampleRates[i]}_StartGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", $@"{trial}\{sampleRates[i]}_{c}_GtDisplacements", c);

            }
            AtomGraph[] TargetGraphs = Helper.ArrayOfFunction(i => new AtomGraph($@"{trial}\{sampleRates[i]}_{c}_TarGraph.graph", TarIms[i]), steps);


            #region FirstStep
            if (!Directory.Exists($@"{trial}\StepOne"))
            {
                Directory.CreateDirectory($@"{trial}\StepOne");
            }
            TarIms[0].WriteMRC($@"{trial}\StepOne\{sampleRates[0]}_{c}_TarIm.mrc");


            DateTime begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) =>
            {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[0]}_StartGraph.graph", TarIms[0]);
                        for (; i < numIterations[0]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }

            }, null);
            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion

            /* Evaluate first step */

            double minDisplRMSD = double.MaxValue;

            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[0].AsConvolvedGaussian(1));
                        double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[0].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[0] = corrScale;
                            minDistScales[0] = distScale;
                            minNormalizings[0] = normalizing;
                            minDisplRMSD = displRMSD;
                        }
                    }
                }
            }
            /* */


            #region SecondStep
            String fromFirstGraphFileStart = $@"{trial}\{sampleRates[0]}_StartGraph.graph";
            AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarMasks[0].AsConvolvedGaussian(1));
            String fromFirstGraphFileFinal = $@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{c}_{minCorrScales[0]}_{minDistScales[0]}_{minNormalizings[0]}_final.graph";
            AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarMasks[0].AsConvolvedGaussian(1));

            List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
            for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{trial}\StepTwo"))
            {
                Directory.CreateDirectory($@"{trial}\StepTwo");
            }
            begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) =>
            {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[1]}_StartGraph.graph", TarIms[1]);
                        localStartGraph.setPositions(fromFirstGraphStart, displacements);
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.graph");
                        for (; i < numIterations[1]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                        
                    }

                }

            }, null);
            end = DateTime.UtcNow;
            System.Console.WriteLine($"Second Step Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion
            /* Evaluate second step */

            minDisplRMSD = double.MaxValue;

            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[1].AsConvolvedGaussian(1));
                        double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[1].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[1] = corrScale;
                            minDistScales[1] = distScale;
                            minNormalizings[1] = normalizing;
                        }
                    }
                }
            }
            /* */


            #region ThirdStep
            String fromSecondGraphFileStart = $@"{trial}\{sampleRates[1]}_StartGraph.graph";
            AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarMasks[1].AsConvolvedGaussian(1));
            String fromSecondGraphFileFinal = $@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{c}_{minCorrScales[1]}_{minDistScales[1]}_{minNormalizings[1]}_final.graph";
            AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarMasks[1].AsConvolvedGaussian(1));

            displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
            for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{trial}\StepThree"))
            {
                Directory.CreateDirectory($@"{trial}\StepThree");
            }
            begin = DateTime.UtcNow;
            Helper.ForCPU(0, 20, 20, null, (k, id, ts) =>
            {
                float corrScale = corrScales[k];

                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[2]}_StartGraph.graph", TarIms[2]);
                        localStartGraph.setPositions(fromSecondGraphStart, displacements);
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.graph");
                        for (; i < numIterations[2]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing);
                            localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.graph");
                        }
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }

            }, null);
            end = DateTime.UtcNow;
            System.Console.WriteLine($"Third Step Total Elapsed Time: {(end - begin).Milliseconds} ms");
            #endregion
            /* Evaluate third step */

            minDisplRMSD = double.MaxValue;

            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{c}_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[2].AsConvolvedGaussian(1));
                        double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[2].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[2] = corrScale;
                            minDistScales[2] = distScale;
                            minNormalizings[2] = normalizing;
                        }
                    }
                }
            }
            /* */


            return;
        }

        

        public static void GridSearchParams(String startImFile, String startMaskFile, String tarImFile, String tarMaskFile, String StartGraph, String TargetGraph, String GtDisplacements, String trial, int targetCount = 2000)
        {
            if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            const int steps = 3;
            float[] sampleRates = new float[steps] { 4, 2, 1 };
            int[] numIterations = new int[steps] { 30, 10, 10 };
            int[] sampledCounts = Helper.ArrayOfFunction(i => (int)(targetCount / Math.Pow(sampleRates[i],3)), steps);

            float[] corrScales = new float[1] { 1 };// Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            bool[] normalizings = new bool[] { /*true,*/ false };

            float[] minCorrScales = new float[steps];
            float[] minDistScales = new float[steps];
            bool[] minNormalizings = new bool[steps];


            Image startIm = Image.FromFile(startImFile);
            ImageProcessor.Normalize01(startIm);
            Image startMask = Image.FromFile(startMaskFile);
            Image tarIm = Image.FromFile(tarImFile);
            Image tarMask = Image.FromFile(tarMaskFile);



            Image[] StartIms = Helper.ArrayOfFunction(i => true ? startIm.GetCopy() : ImageProcessor.Downsample(startIm, sampleRates[i]), steps);
            Image[] StartMasks = Helper.ArrayOfFunction(i => true ? startMask.GetCopy() : ImageProcessor.Downsample(startMask, sampleRates[i]), steps);

            AtomGraph[] StartGraphs = Helper.ArrayOfFunction(i => sampleRates[i]==1 ? new AtomGraph(StartGraph, startIm):new AtomGraph(StartIms[i], StartMasks[i], sampledCounts[i]), steps);

            Image[] TarIms = Helper.ArrayOfFunction(i => tarIm.GetCopy(), steps);
            //TarIms[0].WriteMRC($@"{trial}\{sampleRates[0]}_TarIm.mrc");
            Image[] TarMasks = Helper.ArrayOfFunction(i => tarMask.GetCopy(), steps);

            String line;
            System.IO.StreamReader file = new System.IO.StreamReader(GtDisplacements);
            List<float3> gtDisplList = new List<float3>();

            while ((line = file.ReadLine()) != null)
            {
                float[] result = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries).Select(s => float.Parse(s, CultureInfo.InvariantCulture.NumberFormat)).ToArray();
                gtDisplList.Add(new float3(result[0], result[1], result[2]));
            }

            //AtomGraph[] TargetGraphs = Helper.ArrayOfFunction(i => new AtomGraph(TarIms[i], TarMasks[i], sampledCounts[i]), steps);
            AtomGraph[] TargetGraphs = Helper.ArrayOfFunction(i =>
            {
                if (sampleRates[i] == 1)
                {
                    return new AtomGraph(TargetGraph, TarIms[i]);
                }
                AtomGraph tmp = new AtomGraph(StartIms[i], StartMasks[i], sampledCounts[i]);
                tmp.setPositions(StartGraphs[steps-1], gtDisplList);
                return tmp;
            }, steps);
            for (int i = 0; i < steps; i++)
            {
                //StartIms[i] = StartGraphs[i].Repr(1.0);
                //TarIms[i] = TargetGraphs[i].Repr(1.0);
                /*StartIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartIm.mrc");
                TarIms[i].WriteMRC($@"{trial}\{sampleRates[i]}_TarIm.mrc");

                StartMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_StartMask.mrc");
                TarMasks[i].WriteMRC($@"{trial}\{sampleRates[i]}_TarMask.mrc");
                */
                StartGraphs[i].save($@"{trial}\{sampleRates[i]}_StartGraph.xyz");
                StartGraphs[i].save($@"{trial}\{sampleRates[i]}_StartGraph.graph");
                StartGraphs[i].Repr(1.0).WriteMRC($@"{trial}\{sampleRates[i]}_StartGraph.mrc");

                TargetGraphs[i].save($@"{trial}\{sampleRates[i]}_TargetGraph.xyz");
                TargetGraphs[i].save($@"{trial}\{sampleRates[i]}_TargetGraph.graph");
                TargetGraphs[i].Repr(1.0).WriteMRC($@"{trial}\{sampleRates[i]}_TargetGraph.mrc");
            }
            


            #region FirstStep
            if (!Directory.Exists($@"{trial}\StepOne"))
            {
                Directory.CreateDirectory($@"{trial}\StepOne");
            }


            DateTime begin = DateTime.UtcNow;
            /*Helper.ForCPU(0, corrScales.Count(), 20, null, (k, id, ts) =>
            {*/
            foreach (var corrScale in corrScales)
            {
                //float corrScale = corrScales[k];

                //foreach (var distScale in distScales)
                Helper.ForCPU(0, distScales.Count(), 20, null, (d, id, ts) =>
                {
                    float distScale = distScales[d];
                foreach (var normalizing in normalizings)
                    {
                        if (File.Exists($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.xyz"))
                            continue;
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[0]}_StartGraph.graph", StartIms[0]);
                        localStartGraph.setEMIntensities(TarIms[0]);
                        for (; i < numIterations[0]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing, 0.3f*sampleRates[0]);
                            localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.xyz");
                        }
                        localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.xyz");
                        localStartGraph.save($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }, null);

            }//, null);
            DateTime end = DateTime.UtcNow;
            System.Console.WriteLine($"1st Step Total Elapsed Time: {(end - begin).Hours} h {(end - begin).Minutes} m {(end - begin).Seconds} s {(end - begin).Milliseconds} ms");
            #endregion

            /* Evaluate first step */

            double minDisplRMSD = double.MaxValue;
            double maxAgreement = double.MinValue;
            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[0]);
                        if (!File.Exists($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.mrc"))
                        {
                            localGraph.Repr(1.0d, true).WriteMRC($@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.mrc");
                        }
                        /*double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[0].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[0] = corrScale;
                            minDistScales[0] = distScale;
                            minNormalizings[0] = normalizing;
                            minDisplRMSD = displRMSD;
                        }
                        */
                        Image tmp = new Image(TarIms[0].Dims);
                        float[][] currentAtomSpread = tmp.GetHost(Intent.Write);
                        double agreement = localGraph.getCurrentAgreement(currentAtomSpread);
                        if (agreement > maxAgreement)
                        {
                            minCorrScales[0] = corrScale;
                            minDistScales[0] = distScale;
                            minNormalizings[0] = normalizing;
                            maxAgreement = agreement;
                        }
                        tmp.Dispose();
                    }
                }
            }
            /* */


            #region SecondStep
            String fromFirstGraphFileStart = $@"{trial}\{sampleRates[0]}_StartGraph.xyz";
            AtomGraph fromFirstGraphStart = new AtomGraph(fromFirstGraphFileStart, TarIms[0]);
            String fromFirstGraphFileFinal = $@"{trial}\StepOne\{sampleRates[0]}_Rotate_PI_{minCorrScales[0]}_{minDistScales[0]}_{minNormalizings[0]}_final.xyz";
            AtomGraph fromFirstGraphFinal = new AtomGraph(fromFirstGraphFileFinal, TarIms[0]);

            List<float3> displacements = new List<float3>(fromFirstGraphFinal.Atoms.Count);
            for (int j = 0; j < fromFirstGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromFirstGraphFinal.Atoms[j].Pos - fromFirstGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{trial}\StepTwo"))
            {
                Directory.CreateDirectory($@"{trial}\StepTwo");
            }
            begin = DateTime.UtcNow;
            //Helper.ForCPU(0, corrScales.Count(), 20, null, (k, id, ts) =>
            foreach(var corrScale in corrScales)
            {
                //float corrScale = corrScales[k];

                //foreach (var distScale in distScales)
                Helper.ForCPU(0, distScales.Count(), 20, null, (d, id, ts) =>
                {
                    float distScale = distScales[d];

                    foreach (var normalizing in normalizings)
                    {
                        if (File.Exists($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph"))
                            continue;
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[1]}_StartGraph.graph", StartIms[1]);
                        localStartGraph.setPositions(fromFirstGraphStart, displacements);
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.xyz");
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.graph");
                        localStartGraph.setEMIntensities(TarIms[1]);
                        for (; i < numIterations[1]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing, 0.1f * sampleRates[1]);
                            localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.xyz");
                        }
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.xyz");
                        localStartGraph.save($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }, null);

            }//, null);
            end = DateTime.UtcNow;
            System.Console.WriteLine($"Second Step Total Elapsed Time: {(end - begin).Hours} h {(end - begin).Minutes} m {(end - begin).Seconds} s {(end - begin).Milliseconds} ms");
            #endregion
            /* Evaluate second step */

            minDisplRMSD = double.MaxValue;
            maxAgreement = double.MinValue;
            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[1]);
                        if (!File.Exists($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.mrc"))
                        {
                            localGraph.Repr(1.0d, true).WriteMRC($@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.mrc");
                        }
                        /*double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[1].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[1] = corrScale;
                            minDistScales[1] = distScale;
                            minNormalizings[1] = normalizing;
                        }
                        */
                        float[][] currentAtomSpread = new Image(TarIms[1].Dims).GetHost(Intent.Write);
                        double agreement = localGraph.getCurrentAgreement(currentAtomSpread);
                        if (agreement > maxAgreement)
                        {
                            minCorrScales[1] = corrScale;
                            minDistScales[1] = distScale;
                            minNormalizings[1] = normalizing;
                            maxAgreement = agreement;
                        }
                    }
                }
            }
            /* */


            #region ThirdStep
            String fromSecondGraphFileStart = $@"{trial}\{sampleRates[1]}_StartGraph.graph";
            AtomGraph fromSecondGraphStart = new AtomGraph(fromSecondGraphFileStart, TarIms[1]);
            String fromSecondGraphFileFinal = $@"{trial}\StepTwo\{sampleRates[1]}_Rotate_PI_{minCorrScales[1]}_{minDistScales[1]}_{minNormalizings[1]}_final.graph";
            AtomGraph fromSecondGraphFinal = new AtomGraph(fromSecondGraphFileFinal, TarIms[1]);

            displacements = new List<float3>(fromSecondGraphFinal.Atoms.Count);
            for (int j = 0; j < fromSecondGraphStart.Atoms.Count; j++)
            {
                displacements.Add(fromSecondGraphFinal.Atoms[j].Pos - fromSecondGraphStart.Atoms[j].Pos);
            }



            if (!Directory.Exists($@"{trial}\StepThree"))
            {
                Directory.CreateDirectory($@"{trial}\StepThree");
            }
            begin = DateTime.UtcNow;
            //Helper.ForCPU(0, corrScales.Count(), 20, null, (k, id, ts) =>
            foreach (var corrScale in corrScales)
            {
                //float corrScale = corrScales[k];
                //foreach (var distScale in distScales)
                Helper.ForCPU(0, distScales.Count(), 20, null, (d, id, ts) =>
                {
                    float distScale = distScales[d];
                    foreach (var normalizing in normalizings)
                    {
                        if (File.Exists($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph"))
                            continue;
                        int i = 0;
                        AtomGraph localStartGraph = new AtomGraph($@"{trial}\{sampleRates[2]}_StartGraph.graph", StartIms[2]);
                        localStartGraph.setPositions(fromSecondGraphStart, displacements);
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.xyz");
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{0}.graph");
                        localStartGraph.setEMIntensities(TarIms[2]);
                        for (; i < numIterations[2]; i++)
                        {
                            localStartGraph.moveAtoms(corrScale, distScale, normalizing, 0.1f * sampleRates[2]);
                            localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_it{i + 1}.xyz");
                        }
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.xyz");
                        localStartGraph.save($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph");
                    }

                }, null);

            }//, null);
            end = DateTime.UtcNow;
            System.Console.WriteLine($"Third Step Total Elapsed Time: {(end - begin).Hours} h {(end - begin).Minutes} m {(end - begin).Seconds} s {(end - begin).Milliseconds} ms");
            #endregion
            /* Evaluate third step */

            minDisplRMSD = double.MaxValue;
            maxAgreement = double.MinValue;
            foreach (var corrScale in corrScales)
            {
                foreach (var distScale in distScales)
                {
                    foreach (var normalizing in normalizings)
                    {
                        AtomGraph localGraph = new AtomGraph($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.graph", TarIms[2]);
                        if (!File.Exists($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.mrc"))
                        {
                            localGraph.Repr(1.0d, true).WriteMRC($@"{trial}\StepThree\{sampleRates[2]}_Rotate_PI_{corrScale:#.#}_{distScale:#.#}_{normalizing}_final.mrc");
                        }
                        /*double displRMSD = 0.0;
                        for (int i = 0; i < localGraph.Atoms.Count(); i++)
                        {
                            float3 dist = TargetGraphs[2].Atoms[i].Pos - localGraph.Atoms[i].Pos;
                            displRMSD += Math.Pow(dist.X, 2) + Math.Pow(dist.Y, 2) + Math.Pow(dist.Z, 2);
                        }
                        displRMSD = Math.Sqrt(displRMSD);
                        if (displRMSD < minDisplRMSD)
                        {
                            minCorrScales[2] = corrScale;
                            minDistScales[2] = distScale;
                            minNormalizings[2] = normalizing;
                        }*/
                        Image tmp = new Image(TarIms[2].Dims);
                        float[][] currentAtomSpread = tmp.GetHost(Intent.Write);
                        double agreement = localGraph.getCurrentAgreement(currentAtomSpread);
                        if (agreement > maxAgreement)
                        {
                            minCorrScales[2] = corrScale;
                            minDistScales[2] = distScale;
                            minNormalizings[2] = normalizing;
                            maxAgreement = agreement;
                        }
                        tmp.Dispose();
                    }
                }
            }
            /* */


            return;
        }


        static void shift2points()
        {
            String trial = "Shift2Points";
            /* Create Graphs */
            Image pointIm = Image.FromFile("2PointVolume_Created.mrc");
            Image pointShiftedIm = Image.FromFile("2PointVolumeShifted_Created.mrc");
            Image pointMask = pointIm.GetCopy();
            pointMask.Binarize((float)(1.0f / Math.E));
            Image pointShiftedMask = pointShiftedIm.GetCopy();
            pointShiftedMask.Binarize((float)(1.0f / Math.E));
            //AtomGraph pointGraph = new AtomGraph(pointShiftedIm, pointMask, 2);
            //AtomGraph pointShiftedGraph = new AtomGraph(pointShiftedIm, pointShiftedMask, 2);

            AtomGraph pointGraph = new AtomGraph(pointShiftedIm, pointMask, 2);
            AtomGraph pointShiftedGraph = new AtomGraph(pointShiftedIm, pointShiftedMask, 2);

            /* iterate Movements */

            int i = 0;

            pointShiftedGraph.Repr().WriteMRC($"{trial}_Target.mrc");
            pointGraph.Repr().WriteMRC($"{trial}_Start.mrc");

            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(50));
            for (; i < 20; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(35));
            for (; i < 40; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(20));
            for (; i < 60; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm.AsConvolvedGaussian(10));
            for (; i < 80; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
            pointGraph.setEMIntensities(pointShiftedIm);
            for (; i < 100; i++)
            {
                pointGraph.moveAtoms();
                pointGraph.Repr().WriteMRC($"{trial}_im_it{i + 1}.mrc");
            }
        }

        public static void graphTest()
        {
            String trial = "TestUpscaling"; if (!Directory.Exists(trial))
            {
                Directory.CreateDirectory(trial);
            }
            Image downsampledIm = Image.FromFile(@"downsampling\Downsampled_4.mrc");
            AtomGraph downsampled = new AtomGraph($@"downsampling\StartGraph_downSampled_4.graph", downsampledIm);

            Image nextIm = Image.FromFile(@"downsampling\Downsampled_2.mrc");
            AtomGraph nextGraph = new AtomGraph($@"downsampling\StartGraph_downSampled_2.graph", nextIm);

            downsampled.Repr().WriteMRC($@"{trial}\DownScaledGraphStart.mrc");
            rotateGraph($@"downsampling\StartGraph_downSampled_4.graph", $@"{trial}\StartGraph_downSampled_4_rotate_5.graph", $@"{trial}\StartGraph_downSampled_4_rotate_5.displacements.txt", 5);
            downsampled = new AtomGraph($@"{trial}\StartGraph_downSampled_4_rotate_5.graph", downsampledIm);
            downsampled.Repr().WriteMRC($@"{trial}\DownScaledGraphUpdated.mrc");

            AtomGraph rotated = new AtomGraph($@"{trial}\StartGraph_downSampled_4_rotate_5.graph", downsampledIm);

            List<float3> displacements = new List<float3>(rotated.Atoms.Count);
            for (int i = 0; i < rotated.Atoms.Count; i++)
            {
                displacements.Add(rotated.Atoms[i].Pos - downsampled.Atoms[i].Pos);
            }
            nextGraph.Repr().WriteMRC($@"{trial}\UpscaledGraphStart.mrc");
            nextGraph.setPositions(downsampled, displacements);
            nextGraph.Repr().WriteMRC($@"{trial}\UpscaledGraphUpdated.mrc");

        }

        static string MapPath = "";
        static string MaskPath = "";
        static string WorkingDirectory = "";
        static int NThreads;
        static int NAtoms;
        static void Main(string[] args)
        {
            Options Options = new Options();
            //if (!Debugger.IsAttached)
            {
                ParserResult<Options> parseRes = Parser.Default.ParseArguments<Options>(args);
                parseRes.WithNotParsed(i =>
                {
                    Environment.Exit(1);
                });
                parseRes.WithParsed<Options>(opts => Options = opts);
                WorkingDirectory = Environment.CurrentDirectory + "/";

                MapPath = Options.MapPath;
                MaskPath = Options.MaskPath;
                NAtoms = Options.NAtoms;
            }

            //String rootDir = @"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy_Modulated\100\";
            //if (!Directory.Exists(rootDir))
            //{
            //    Directory.CreateDirectory(rootDir);
            //}
            //Directory.SetCurrentDirectory(rootDir);


            //int it = 100;
            /*
            Image im = Image.FromFile(@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy\100\trial10000\4_StartIm.mrc");
            ImageProcessor.Normalize01(im);

            Image tarIm = Image.FromFile(@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy\100\trial10000\4_TarIm.mrc");
            ImageProcessor.Normalize01(tarIm);

            Image mask = Image.FromFile(@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy\100\trial10000\4_StartMask.mrc");
            Image tarMask = Image.FromFile(@"D:\Software\FlexibleRefinement\bin\Debug\PulledProtein\Toy\100\trial10000\4_TarMask.mrc");
            */
            //AtomGraph graph = new AtomGraph(im, mask, (int)(10000.0/64.0));
            //AtomGraph tarGraph = new AtomGraph(tarIm, tarMask, (int)(10000.0 / 64.0));

            //Image imFG = tarGraph.Repr(1.0d, true);
            //tarIm.WriteMRC($@"{rootDir}\TargetIm.mrc");
            //graph.setEMIntensities(tarIm);

            //graph.moveAtoms();

            //GridSearchParams(rootDir + @"currentInput\startIm.mrc",
            //    rootDir + $@"currentInput\startMask.mrc",
            //    rootDir + $@"currentInput\TargetIm_fromGraph{it}.mrc",
            //    rootDir + $@"currentInput\TargetMask_fromGraph{it}.mrc",
            //    rootDir + $@"currentInput\startGraph.graph",
            //    rootDir + $@"currentInput\TargetGraph{it}.graph",
            //    rootDir + $@"currentInput\gtDisplacements.txt",
            //    "current_trial10000", 10000);

            //GridSearchParams(8);
            /*for (int c = 8; c < 9; c++)
            {
                doRotationExp(@"D:\Software\FlexibleRefinement\bin\Debug\lennardJones\differentRotExp_No_NeighborUpdate_c10000", c, new List<float>(new float[] { 2, 10, 10 }), new List<float>(new float[] { 17, 13, 13 }), new List<bool>(new bool[] { false, true, true }));
            }*/
            //int c = 10;
            //createRotated(c);
            //String trial = "RotateStick"; if (!Directory.Exists(trial))
            //{
            //    Directory.CreateDirectory(trial);
            //}
            //Image stickIm = Image.FromFile("StickVolume_Created.mrc");
            //Image stickMask = stickIm.GetCopy();
            //stickMask.Binarize((float)(1.0f / Math.E));
            //AtomGraph startGraph = new AtomGraph($@"{trial}\Stick_Initial.graph", stickIm);

            //Image rotated = Image.FromFile($@"{trial}\Rotate_PI_{c}_gt.mrc");
            //rotated.AsConvolvedGaussian(1.0f).WriteMRC($@"{trial}\Rotate_PI_{c}_gt_convolved1.mrc");
            //Image convolved4 = rotated.AsConvolvedGaussian(4.0f);
            //convolved4.WriteMRC($@"{trial}\Rotate_PI_{c}_gt_convolved4.mrc");

            //Image rotatedkMask = convolved4.GetCopy();
            //rotatedkMask.Binarize(0.25f);
            //rotatedkMask.WriteMRC($@"{trial}\Rotate_PI_{c}_gt_convolved4_mask.mrc");
            //startGraph.setEMIntensities(rotatedkMask.AsConvolvedGaussian(5));


            //float[] corrScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            //float[] distScales = Helper.ArrayOfFunction(k => (float)(k + 1), 20);
            //bool[] normalizings = new bool[2] { true, false };
            //Helper.ForCPU(0, 20, 11, null, (k, id, ts) => {
            //    float corrScale = corrScales[k];

            //    foreach (var distScale in distScales)
            //    {
            //        foreach (var normalizing in normalizings)
            //        {
            //            int i = 0;
            //            AtomGraph localStartGraph = new AtomGraph($@"{trial}\Stick_Initial.graph", stickIm);
            //            localStartGraph.setEMIntensities(rotatedkMask.AsConvolvedGaussian(5));
            //            for (; i < 5; i++)
            //            {
            //                localStartGraph.moveAtoms(corrScale, distScale, normalizing);
            //                localStartGraph.Repr().WriteMRC($@"{trial}\Rotate_PI_{c}_im_it{i + 1}_{corrScale:#.#}_{distScale:#.#}_{normalizing}.mrc");
            //            }
            //            localStartGraph.save($@"{trial}\Rotate_PI_{c}_final_{corrScale:#.#}_{distScale:#.#}_{normalizing}.graph");
            //        }

            //    }

            //}, null);
            //startGraph.setEMIntensities(rotatedkMask.AsConvolvedGaussian(1));

            ///*for (; i < 10; i++)
            //{
            //    startGraph.moveAtoms(10.0f, 4.0f,true);
            //    startGraph.Repr().WriteMRC($@"{trial}\Rotate_PI_{c}_im_it{i + 1}.mrc");
            //}*/
            //startGraph.save($@"{trial}\Rotate_PI_{c}_final.graph");
        }
    }
}
