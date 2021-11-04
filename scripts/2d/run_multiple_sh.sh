#! /bin/sh
num_times=30
seed=0
for i in `seq 1 $num_times`
do
    echo "Running $i iteration seed $seed..."
    sh ./scripts/2d/mopa_gen_data.sh 0 $seed
    echo "Finished $i iteration seed $seed..."
    seed=$((seed+2))
done