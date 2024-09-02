#!/bin/bash
data=house_16H
tree_depth=10
train_data_count=10000
scale=1G

python train_dt.py -d $data -td $tree_depth -dc $train_data_count
model_name=$(cat ./model/model_name.txt)

python run_onnx.py -d $data -s $scale -m $model_name -p 0

while read predicate
do
    python pruning.py -m $model_name -p $predicate
    python run_onnx.py -d $data -s $scale -m $model_name -p $predicate --pruned
done < ./model/model_leaf_range.txt
