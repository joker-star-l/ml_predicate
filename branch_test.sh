for((i=0;i<=100;i++));  
do   
echo $i;  
python branch_test.py -d $i -l
python branch_test.py -d $i
done
