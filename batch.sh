#!/bin/bash

# thread-unsafe

# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d house_16H -l price -b onnxruntime
./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount -b onnxruntime
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 10G -t 1 -d Ailerons -l goal -b onnxruntime

# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d house_16H -l price -b sklearn
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount -b sklearn
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d Ailerons -l goal -b sklearn

# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d house_16H -l price -b sql_duckdb
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount -b sql_duckdb
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d Ailerons -l goal -b sql_duckdb

# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 10G -t 1 -d house_16H -l price -b lleaves
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount -b lleaves
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 10G -t 1 -d Ailerons -l goal -b lleaves

# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d house_16H -l price -b treelite
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount -b treelite
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d Ailerons -l goal -b treelite

# ./generate_and_run_single_model_clf.sh -td 10 -dc 10000 -s 1G -t 1 -d bank-marketing -l Class -b onnxruntime

# ./generate_and_run_single_model_clf.sh -td 3 -dc 10000 -s 1G -t 1 -d california -l price_above_median -b onnxruntime

# ./generate_and_run_single_model_clf.sh -td 3 -dc 10000 -s 1G -t 1 -d electricity -l class -b onnxruntime

# ./generate_and_run_single_model_clf.sh -td 3 -dc 10000 -s 1G -t 1 -d credit -l SeriousDlqin2yrs -b onnxruntime

# ./generate_and_run_single_model_clf.sh -td 10 -dc 10000 -s 1G -t 1 -d NASA -l hazardous -b onnxruntime

# motivation: nyc-taxi-green-dec-2016_d3_l8_n15_20241105114517
# ./generate_and_run_single_model.sh -td 3 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount -b sklearn

# dtcp: bank-marketing_d3_l8_n15_20241104161453

# ./generate_and_run_single_model.sh -td 5 -dc 10000 -s 1G -t 1 -d house_16H -l price -b sklearn