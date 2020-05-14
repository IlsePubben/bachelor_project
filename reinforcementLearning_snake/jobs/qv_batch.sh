#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=qv_batch 
#SBATCH --output=qv_batch
#SBATCH --mem=800
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --array=1-10
module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow

python main.py -a qv-learning -e 0.1 -y 0.99 --lrQ 0.0005 --lrV 0.0005 --lrA 0.0005 -v 7
