#!/bin/bash

# Script to run the neural matching algorithm on random and real-world networks.

alpha=("0.001")
beta=("1.0")

exe=./neural_matching.py # version to run.

#========================================================
#                     
#========================================================
echo "running 1-to-many random.."

Ns=("500")
Ms=("550" "1000" "2000" "5000")
deg=complete
weight=("lognormal" "poisson" "uniform")

for w in "${weight[@]}"; do
    for a in "${alpha[@]}"; do
        for b in "${beta[@]}"; do
            for N in "${Ns[@]}"; do
                for M in "${Ms[@]}"; do

                    $exe -N $N -M $M -d $deg -w $w -a $a -b $b > ../results/${deg}_${w}_n${N}_m${M}_a${a}_b${b}.txt &

                done

            done
        done
    done
done

echo "running 1-to-1 random.."

Ns=("250" "500" "1000")
deg=complete
weight=("lognormal" "poisson" "uniform")

for w in "${weight[@]}"; do
    for a in "${alpha[@]}"; do
        for b in "${beta[@]}"; do
            for N in "${Ns[@]}"; do

                # Note: we set N = M here! 
                $exe -N $N -M $N -d $deg -w $w -a $a -b $b > ../results/${deg}_${w}_n${N}_m${N}_a${a}_b${b}.txt &

            done
        done
    done
done


echo "running real-world.."

FILES=("abtbuy" "assignp5000" "celegans" "movielens" "uiuc" "yaron" "jessey2r")

for f in "${FILES[@]}"; do
    for a in "${alpha[@]}"; do
        for b in "${beta[@]}"; do

            $exe -f $f -a $a -b $b -p > ../results/${f}_a${a}_b${b}.txt &

        done
    done
done

wait

echo "done"
