#!/bin/bash

# thread-unsafe 

# data=house_16H
# label=price

data=nyc-taxi-green-dec-2016
label=tipamount

# data=Ailerons
# label=goal

tree_depth=10
train_data_count=10000
scale=1G
threads=2

python train_dt.py -d $data -td $tree_depth -dc $train_data_count -l $label
model_name=$(cat ./model/model_name.txt)

python run_onnx.py -d $data -s $scale -m $model_name -p 0 -t $threads

while read predicate
do
    python pruning.py -m $model_name -p $predicate
    python rotation.py -m $model_name
    # python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads
    python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
    python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
    
    # break
done < ./model/model_leaf_range.txt
