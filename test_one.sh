model_name="nyc-taxi-green-dec-2016_d10_l858_n1715_20241010162144"
predicate="0.850889"
data="nyc-taxi-green-dec-2016"
scale="1G"
threads=2

# python pruning.py -m $model_name -p $predicate
# python onnx2sklearn.py -m $model_name --pruned 1
# python dp.py -m $model_name
# python onnx2sklearn.py -m $model_name --pruned 2

# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2

python onnx2sql.py -m $model_name --pruned 0
python pruning.py -m $model_name -p $predicate
python onnx2sql.py -m $model_name --pruned 1
python dp.py -m $model_name
python onnx2sql.py -m $model_name --pruned 2

python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0
python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2