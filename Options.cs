using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommandLine;
namespace FlexibleRefinement
{
    class Options
    {

        [Option('i', Required = true, HelpText = "Path to a mrc file containing the EM map to be estimated.")]
        public string MapPath { get; set; }

        [Option('m', Required = true, HelpText = "Number of threads used for calculations.")]
        public string MaskPath { get; set; }


        [Option("threads", Default = 5, HelpText = "Number of threads used for calculations.")]
        public int NThreads { get; set; }

        [Option("NAtoms", Default = 1000, HelpText = "Number of threads used for calculations.")]
        public int NAtoms { get; set; }


    }
}
