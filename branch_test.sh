for((i=0;i<=50;i++));  
do   
echo $i;  
python branch_test.py -d $i -l
python branch_test.py -d $i
done
