#!/bin/bash

# thread-unsafe 

data=nyc-taxi-green-dec-2016
tree_depth=10
train_data_count=10000
label=tipamount
scale=1G
threads=1

python train_dt.py -d $data -td $tree_depth -dc $train_data_count -l $label
model_name=$(cat ./model/model_name.txt)

python run_onnx.py -d $data -s $scale -m $model_name -p 0 -t $threads

while read predicate
do
    python pruning.py -m $model_name -p $predicate
    python run_onnx.py -d $data -s $scale -m $model_name -p $predicate --pruned -t $threads
done < ./model/model_leaf_range.txt
