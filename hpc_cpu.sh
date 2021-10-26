#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=8192  # memory in Mb
#SBATCH -t 15:00:00  # time requested in hour:minute:second

# Prevents `plt.show()` for attempting connecting.
unset DISPLAY XAUTHORITY 

cd ~/overparameterization
source activate overparameterization
echo "... Job beginning"
"$@"
echo "... Job Ended"
