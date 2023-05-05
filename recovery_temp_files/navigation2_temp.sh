#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..1}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name navigation2 --use_recovery --MF_recovery --gamma_safe 0.65 --eps_safe 0.2 --logdir navigation2 --logdir_suffix RRL_MF --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done