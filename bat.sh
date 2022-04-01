#!/bin/bash

CONFIG_FOLD=example_configurations/bat

for file in $CONFIG_FOLD/*; do
  CONFIG_NO_EXT="${file%.*}"
  OUTPUT_FOLD=output_benchmarks/$(basename $CONFIG_NO_EXT)
  echo python3 ./run.py -c $file -o $OUTPUT_FOLD
  python3 ./run.py -c $file -o $OUTPUT_FOLD
done
