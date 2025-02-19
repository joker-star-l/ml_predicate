# model_name="nyc-taxi-green-dec-2016_t3_d2_l4_n7_20250204160726"
# predicate="1.275"
# data="nyc-taxi-green-dec-2016"
# scale="1G"
# threads=1

# model_name="nyc-taxi-green-dec-2016_t10_d10_l849_n1698_20250209144203"
# predicate="1.275"
# data="nyc-taxi-green-dec-2016"
# scale="1G"
# threads=4

# python run_onnx_rf.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0
# python run_onnx_rf.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1


# model_name="bank-marketing_t3_d2_l4_n7_20250209151527"
# predicate="1"
# data="bank-marketing"
# scale="1G"
# threads=1

model_name="bank-marketing_t10_d10_l318_n635_20250210050038"
predicate="1"
data="bank-marketing"
scale="1G"
threads=4

python run_onnx_rf.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0 --clf
python run_onnx_rf.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1 --clf

# python pruning.py -m $model_name -p $predicate
# python onnx2sklearn.py -m $model_name --pruned 1
# python dp.py -m $model_name
# python onnx2sklearn.py -m $model_name --pruned 2
# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2

# python onnx2sql.py -m $model_name --pruned 0
# python pruning.py -m $model_name -p $predicate
# python onnx2sql.py -m $model_name --pruned 1
# python dp.py -m $model_name
# python onnx2sql.py -m $model_name --pruned 2
# python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0
# python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_sql_duckdb.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2

# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0
# python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0

# python onnx2lgb.py -m $model_name --pruned 0
# python pruning.py -m $model_name -p $predicate
# python onnx2lgb.py -m $model_name --pruned 1
# python dp.py -m $model_name
# python onnx2lgb.py -m $model_name --pruned 2
# python run_lleaves.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0
# python run_lleaves.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_lleaves.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2

# python onnx2lgb.py -m $model_name --pruned 0 --genc
# python pruning.py -m $model_name -p $predicate
# python onnx2lgb.py -m $model_name --pruned 1 --genc
# python dp.py -m $model_name
# python onnx2lgb.py -m $model_name --pruned 2 --genc
# python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0
# python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2

# python run_onnx.py -d $data -s $scale -m $model_name -p 0 -t $threads --clf
# python run_onnx.py -d $data -s $scale -m $model_name -p 0 -t $threads --clf --clf2reg

# python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_onnx.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
# python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2
