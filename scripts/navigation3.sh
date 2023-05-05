#!/bin/bash

# Recovery RL (model-free recovery)
for i in {1..5}
do
	echo "RRL MF Run $i"
	python -m rrl_main --cuda --env-name navigation3 --use_recovery --MF_recovery --gamma_safe 0.65 --eps_safe 0.2 --logdir navigation3 --logdir_suffix RRL_MF --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done

# Unconstrained
for i in {1..5}
do
	echo "Unconstrained Run $i"
	python -m rrl_main --env-name navigation3 --cuda --logdir navigation3 --logdir_suffix unconstrained --num_eps 400 --num_unsafe_transitions 20000 --seed $i
done