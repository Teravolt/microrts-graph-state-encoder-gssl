#!/bin/bash

set -x

# Baselines + DINO

# for i in 6 7 8 9 10; do

#     echo "Running seed $i"

#     # Supervised (No GNN)
#     python train_action_prediction_no_gnn.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-supervised-no-gnn-seed-$i.pt --run_name action-predictor-supervised-no-gnn-seed-$i --seed $i

#     python eval_action_prediction_no_gnn.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-supervised-no-gnn-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-supervised-no-gnn-seed-$i --seed $i

#     ## Supervised
#     python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-supervised-v2-encoder-seed-$i.pt --run_name action-predictor-supervised-v2-encoder-seed-$i --seed $i

#     python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-supervised-v2-encoder-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-supervised-v2-encoder-seed-$i --seed $i

#     ## Linear Probe
#     python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-linear-probe-v2-encoder-seed-$i.pt --run_name action-predictor-linear-probe-v2-encoder-seed-$i --state_model gnn-state-model-v2.pt --seed $i

#     python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-linear-probe-v2-encoder-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-linear-probe-v2-encoder-seed-$i --seed $i

#     ## Fine-Tune
#     python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-seed-$i.pt --run_name action-predictor-fine-tune-v2-encoder-seed-$i --state_model gnn-state-model-v2.pt --fine_tune --seed $i

#     python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-seed-$i --seed $i

# done

# SSL Baselines
# for i in 1 2 3 4 5 6 7 8 9 10; do

#     echo "Running seed $i"

#     ## Graph Barlow Twins - Linear Probe
#     python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-linear-probe-v2-encoder-barlow-twins-seed-$i.pt --run_name action-predictor-linear-probe-v2-encoder-barlow-twins-seed-$i --state_model gnn-state-model-v2-barlow-twins.pt --seed $i

#     python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-linear-probe-v2-encoder-barlow-twins-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-linear-probe-v2-encoder-barlow-twins-seed-$i --seed $i

#     ## Graph Barlow Twins - Fine-Tune
#     python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-barlow-twins-seed-$i.pt --run_name action-predictor-fine-tune-v2-encoder-barlow-twins-seed-$i --state_model gnn-state-model-v2-barlow-twins.pt --fine_tune --seed $i

#     python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-barlow-twins-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-barlow-twins-seed-$i --seed $i

#     ## BGRL - Linear Probe
#     python train_action_prediction_bgrl_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-linear-probe-v2-encoder-bgrl-seed-$i.pt --run_name action-predictor-linear-probe-v2-encoder-bgrl-seed-$i --state_model gnn-state-model-v2-bgrl.pt --seed $i

#     python eval_action_prediction_bgrl_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-linear-probe-v2-encoder-bgrl-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-linear-probe-v2-encoder-bgrl-seed-$i --seed $i

#     ## BGRL - Fine-Tune
#     python train_action_prediction_bgrl_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-bgrl-seed-$i.pt --run_name action-predictor-fine-tune-v2-encoder-bgrl-seed-$i --state_model gnn-state-model-v2-bgrl.pt --fine_tune --seed $i

#     python eval_action_prediction_bgrl_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-bgrl-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-bgrl-seed-$i --seed $i

# done

## DINO with centering
for i in 1 2 3 4 5 6 7 8 9 10; do

    echo "Running seed $i"

    ## Linear Probe
    python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-linear-probe-v2-encoder-centering-seed-$i.pt --run_name action-predictor-linear-probe-v2-encoder-centering-seed-$i --state_model gnn-state-model-v2-with-centering.pt --seed $i

    python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-linear-probe-v2-encoder-centering-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-linear-probe-v2-encoder-centering-seed-$i --seed $i

    ## Fine-Tune
    python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-centering-seed-$i.pt --run_name action-predictor-fine-tune-v2-encoder-centering-seed-$i --state_model gnn-state-model-v2-with-centering.pt --fine_tune --seed $i

    python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-centering-seed-$i.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-centering-seed-$i --seed $i

done