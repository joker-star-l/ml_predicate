#!/bin/bash

# thread-unsafe

# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d house_16H -l price
./generate_and_run_single_model.sh -td 10 -dc 10000 -s 1G -t 1 -d nyc-taxi-green-dec-2016 -l tipamount
# ./generate_and_run_single_model.sh -td 10 -dc 10000 -s 10G -t 1 -d Ailerons -l goal
