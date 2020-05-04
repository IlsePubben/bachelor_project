#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=qvamax_learning
#SBATCH --output=qvamax_learning
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ilse.pubben@gmail.com
#SBATCH --mem=800
module load Python/3.6.4-foss-2019a
module load GCCcore/8.2.0
pip install --user matplotlib
pip install --user keras
pip install --user --upgrade tensorflow
for i in {1..5}
do 
    python main.py -a qvamax-learning -e 0.2 -y 0.95 --lrA 0.005
    python main.py -a qvamax-learning -e 0.2 -y 0.95 --lrA 0.0002
done
