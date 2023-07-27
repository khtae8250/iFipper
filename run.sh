#!/bin/bash

mkdir -p results
InitTime=$(date +%s)
# 4.3.1: baseline.py
# TARGET="4.3.1: baseline.py"
# StartTime=$(date +%s)
# for DATASET in 'COMPAS' 'AdultCensus' 'Credit'; do
# for SIM_MAT in 'knn' 'threshold';do
# python baseline.py --dataset $DATASET --similarity_matrix $SIM_MAT --verbose
# done
# done
# EndTime=$(date +%s)
# echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." > /dev/stdout
# echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." >> ./results/out.txt

# # 4.3.2: runtime.py
# TARGET="4.3.2: runtime.py"
# StartTime=$(date +%s)
# for DATASET in 'Synthetic'; do
# for SIM_MAT in 'knn';do
# python runtime.py --dataset $DATASET --similarity_matrix $SIM_MAT --verbose
# done
# done
# EndTime=$(date +%s)
# echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." > /dev/stdout
# echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." >> ./results/out.txt

# 4.4: solution.py
TARGET="4.4: solution.py"
StartTime=$(date +%s)
for DATASET in 'AdultCensus'; do
for SIM_MAT in 'knn';do
python solution.py --dataset $DATASET --similarity_matrix $SIM_MAT --verbose
done
done
EndTime=$(date +%s)
echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." > /dev/stdout
echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." >> ./results/out.txt

# 4.5: ablation.py
TARGET="4.5: ablation.py"
StartTime=$(date +%s)
for DATASET in 'AdultCensus'; do
for SIM_MAT in 'knn';do
python ablation.py --dataset $DATASET --similarity_matrix $SIM_MAT --verbose
done
done
EndTime=$(date +%s)
echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." > /dev/stdout
echo "$(($EndTime - $StartTime)) seconds to complete $TARGET." >> ./results/out.txt

FinalTime=$(date +%s)
echo "Overall task takes $(($EndTime - $InitTime)) seconds." > /dev/stdout
echo "Overall task takes $(($EndTime - $InitTime)) seconds." >> ./results/out.txt
