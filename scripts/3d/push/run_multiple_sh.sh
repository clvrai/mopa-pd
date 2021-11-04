#! /bin/sh
# Please modify mopa_gen_data.sh first before running this script.

num_times=21
seed=0
for i in `seq 1 $num_times`
do
    echo "Running $i iteration seed $seed..."
    sh ./scripts/3d/push/mopa_gen_data.sh 1 $seed
    echo "Finished $i iteration seed $seed..."
    seed=$((seed+2))
done