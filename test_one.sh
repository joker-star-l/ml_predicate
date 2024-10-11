model_name="nyc-taxi-green-dec-2016_d10_l858_n1715_20241010162144"
predicate="1.701778"
data="nyc-taxi-green-dec-2016"
scale="1G"
threads=1

python pruning.py -m $model_name -p $predicate
python onnx2sklearn.py -m $model_name --pruned 1
python dp.py -m $model_name
python onnx2sklearn.py -m $model_name --pruned 2

python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1
python run_sklearn.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2