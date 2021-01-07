#!/bin/bash
# Submission script for Lemaitre3
#SBATCH --job-name={}
#SBATCH --time=0-12:00:00 # days-hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2625 # megabytes
#SBATCH --partition=batch
#
#SBATCH --mail-user=Chris.Adam@ulb.be
#SBATCH --mail-type=ALL
#
#SBATCH --comment=LD_FEN

cd FEN
module load TensorFlow/2.2.0-foss-2019b-Python-3.7.4
python3 main.py {} {} {} {} {} {}