#!/bin/bash

while IFS=, read -r arg1 arg2 arg3 arg4 arg5
do
    python examples/data_creation/main.py --num_scenes "$arg1" --num_ep_per_scene "$arg2" --step_size "$arg3" --turn_angle "$arg4" --save_path "$arg5"
done < data_creation_args.txt