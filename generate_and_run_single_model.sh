#!/bin/bash

# thread-unsafe 

# data=house_16H
# label=price

# data=nyc-taxi-green-dec-2016
# label=tipamount

# data=Ailerons
# label=goal

# tree_depth=10
# train_data_count=10000
# scale=1G
# threads=1

while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--data) data=$2; shift 2;;
        -l|--label) label=$2; shift 2;;
        -td|--tree_depth) tree_depth=$2; shift 2;;
        -dc|--data_count) train_data_count=$2; shift 2;;
        -s|--scale) scale=$2; shift 2;;
        -t|--threads) threads=$2; shift 2;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
done

echo "data: $data"
echo "label: $label"
echo "tree_depth: $tree_depth"
echo "train_data_count: $train_data_count"
echo "scale: $scale"
echo "threads: $threads"

python train_dt.py -d $data -td $tree_depth -dc $train_data_count -l $label
model_name=$(cat ./model/model_name.txt)

python run_onnx.py -d $data -s $scale -m $model_name -p 0 -t $threads

while read predicate
do
    python pruning.py -m $model_name -p $predicate
    python dp.py -m $model_name

    # python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads
    python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
    python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
    
    # break
done < ./model/model_leaf_range.txt
