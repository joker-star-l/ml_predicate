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
        -b|--backend) backend=$2; shift 2;; # onnxruntime, sklearn, sql_duckdb, lleaves, treelite
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

if [ "$backend" = "onnxruntime" ]; then
    python percent_value.py -d $data -l $label -m $model_name -t $threads
    python run_onnx.py -d $data -s $scale -m $model_name -p 0 -t $threads

    while read predicate
    do
        python pruning.py -m $model_name -p $predicate
        python merge.py -m $model_name

        # python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads
        python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
        python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
        
        # break
    done < ./model/${model_name}_percent_value_test.txt

elif [ "$backend" = "sklearn" ]; then
    python run_sklearn.py -d $data -s $scale -m $model_name -p 0 -t $threads

    while read predicate
    do
        python pruning.py -m $model_name -p $predicate
        python onnx2sklearn.py -m $model_name --pruned 1
        python dp.py -m $model_name
        python onnx2sklearn.py -m $model_name --pruned 2

        python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
        python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
        
        # break
    done < ./model/model_leaf_range.txt

elif [ "$backend" = "sql_duckdb" ]; then
    python onnx2sql.py -m $model_name --pruned 0
    python run_sql_duckdb.py -d $data -s $scale -m $model_name -p 0 -t $threads

    while read predicate
    do
        python pruning.py -m $model_name -p $predicate
        python onnx2sql.py -m $model_name --pruned 1
        python dp.py -m $model_name
        python onnx2sql.py -m $model_name --pruned 2

        python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
        python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
        
        # break
    done < ./model/model_leaf_range.txt

elif [ "$backend" = "lleaves" ]; then
    python onnx2lgb.py -m $model_name --pruned 0
    python run_lleaves.py -d $data -s $scale -m $model_name -p 0 -t $threads

    while read predicate
    do
        python pruning.py -m $model_name -p $predicate
        python onnx2lgb.py -m $model_name --pruned 1
        python dp.py -m $model_name
        python onnx2lgb.py -m $model_name --pruned 2

        python run_lleaves.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
        python run_lleaves.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
        
        # break
    done < ./model/model_leaf_range.txt

elif [ "$backend" = "treelite" ]; then
    python onnx2lgb.py -m $model_name --pruned 0 --genc
    python run_treelite.py -d $data -s $scale -m $model_name -p 0 -t $threads

    while read predicate
    do
        python pruning.py -m $model_name -p $predicate
        python onnx2lgb.py -m $model_name --pruned 1 --genc
        python dp.py -m $model_name
        python onnx2lgb.py -m $model_name --pruned 2 --genc

        python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
        python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
        
        # break
    done < ./model/model_leaf_range.txt

else
    echo "Unknown backend: $backend"
    exit 1
fi
