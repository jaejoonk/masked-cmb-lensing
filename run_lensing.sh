#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --constraint=haswell
#SBATCH --qos=interactive
#SBATCH --account=mp107
#SBATCH --mail-user=jaejoonk@sas.upenn.edu
#SBATCH --mail-type=BEGIN,END

module load python/3.7-anaconda-2019.07
srun -n 2 python3 websky_new_lensing_procedure.py
