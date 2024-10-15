model_name="nyc-taxi-green-dec-2016_d10_l859_n1717_20241015054511"
predicate="1.701778"
data="nyc-taxi-green-dec-2016"
scale="1G"
threads=1

file_to_check0="../model/${model_name}_rf.joblib"
if [ -f "$file_to_check0" ]; then
    echo "File $file_to_check0 exists, skipping df2rf execution."
else
    python dt2rf.py -d $data -m $model_name --pruned 0
fi

file_to_check1="../model/${model_name}_out_rf.joblib"
if [ -f "$file_to_check1" ]; then
    echo "File $file_to_check1 exists, skipping df2rf execution."
else
    python dt2rf.py -d $data -m $model_name --pruned 1
fi

file_to_check2="../model/${model_name}_out2_rf.joblib"
if [ -f "$file_to_check2" ]; then
    echo "File $file_to_check2 exists, skipping df2rf execution."
else
    python dt2rf.py -d $data -m $model_name --pruned 2
fi

python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0 --toolchain gcc
python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1 --toolchain gcc
python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2 --toolchain gcc

python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 0 --toolchain clang
python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 1 --toolchain clang
python run_treelite.py -d $data -s $scale -m $model_name -p $predicate -t $threads --pruned 2 --toolchain clang