# Code Repository for Paper "Pretraining Graph State Encoders for microRTS using Graph Self-Supervised Learning"

This repository contain Python scripts and Jupyter notebooks to (1) pretrain the graph state encoder used in the paper, and (2) run the cluster analysis and action prediction experiments contained in the paper.

## Installation

You will need to use a Python version between 3.9 (inclusive) and 3.13 (exclusive) to run the scripts and notebooks.

We use Poetry to set up our Python environment as it provides an easy and quick way to setup, update, and tear down python virtual environments.
Please see the [Poetry Documentation](https://python-poetry.org/) for instructions on how to both install it on your preferred OS and use it.

If not already installed, I would recommend installing [Poetry's shell plugin](https://github.com/python-poetry/poetry-plugin-shell) as it spawns a separate shell for running the code:
```bash
poetry self add poetry-plugin-shell
```

Install all dependencies using Poetry as follows (where `3.9 >= *python-version* < 3.13`):
```bash
poetry env use *python-version* # Sets Python version to use for the project
poetry install # Installs dependencies
```

If you prefer to use other methods to set up a Python environment, please make sure that the Python packages and version from `pyproject.toml` are installed.
There is also a `requirements.txt` generated from `pyproject.toml` to help with installation outside of Poetry.

> Please note that this code was run on MacOS with Python 3.11.10.
This should work on Linux and other Python versions, but I am not 100% sure.
If you have any issues running this on other operating systems or other Python versions, let me know and I'll take a look!

## Hardware

The code does not require any special hardware to run and does not assume you have a specific type of hardware.
All of the code in this repository was run on an M2 Macbook Air.
If you are able to get this running on a GPU, let me know (I'm GPU poor :sweat_smile:)!

## microRTS Datasets

We use replay data from the [CoG 2019](https://sites.google.com/site/micrortsaicompetition/competition-results/2019-cog-results?authuser=0) and [CoG 2020](https://sites.google.com/site/micrortsaicompetition/competition-results/2020-cog-results?authuser=0) microRTS competitions for pretraining and experiments.
This data is not mine and was used with permission from Dr. Santiago Ontanon; please message Dr. Ontanon to get access to the dataset.

## Training a Graph State Encoder

**Distillation with No Labels (DINO)**: The Python script ` train_state_encoder_dino_v2_with_centering.py` can be used to pretrain a graph state encoder using DINO.
The exact command-line arguments we used in the paper are as follows:
```bash
python train_state_encoder_dino_v2_with_centering.py ../microrts-dataset/microrts-cog-2019-standard --read_from_zip --max_replay_length 64 --save_model gnn-state-model-v2-centering.pt --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --run_name pre-training-v2-encoder-centering
```

**Graph Barlow Twins (GBT)**: The Python script ` train_state_encoder_barlow_twins_v2.py` can be used to pretrain a graph state encoder using GBT.
The exact command-line arguments we used in the paper are as follows:
```bash
python train_state_encoder_barlow_twins_v2.py ../microrts-dataset/microrts-cog-2019-standard --read_from_zip --max_replay_length 64 --save_model gnn-state-model-v2-barlow-twins.pt --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --run_name pre-training-v2-encoder-barlow-twins
```

**Bootstraped Graph Latents (BGRL)**: The Python script `train_state_encoder_bgrl_v2.py` can be used to pretrain a graph state encoder using BGRL.
The exact arguments we used in the paper are as follows:
```bash
python train_state_encoder_bgrl_v2.py ../microrts-dataset/microrts-cog-2019-standard --read_from_zip --max_replay_length 64 --save_model gnn-state-model-v2-bgrl.pt --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --run_name pre-training-v2-encoder-bgrl
```

> NOTE: The first argument to each of these scripts is the path to replay data from the first three iterations of the CoG 2019 competition.
This path will change depending on where you stored your replay dataset.

The pretraining runs, exact parameters used in the paper, and pretrained models can be found in our Weights and Biases project: https://wandb.ai/pkthunder/microrts-graph-state-ssl?nw=nwuserpkthunder

## Action Prediction

There are a few Python scripts to train an action prediction model.
In the paper, we trained models for action prediction using different methods; below we provide the methods and their associated training/evaluation scripts. 
The prediction runs, exact parameters used in the paper, and models can be found in our Weights and Biases project: https://wandb.ai/pkthunder/microrts-action-prediction?nw=nwuserpkthunder

### No GNN

Training and evaluation can be done using `train_action_prediction_no_gnn.py` and `eval_action_prediction_no_gnn.py`.

Example usage for training:
```bash
python train_action_prediction_no_gnn.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-supervised-no-gnn-seed-10.pt --run_name action-predictor-supervised-no-gnn-seed-10 --seed 10
```

Example usage for evaluation:
```bash
python eval_action_prediction_no_gnn.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-supervised-no-gnn-seed-10.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-supervised-no-gnn-seed-10 --seed 10
```

### Random Init

Training and evaluation can be done using `train_action_prediction_v2.py` and `train_action_prediction_v2.py`.

Example usage for training:
```bash
python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-supervised-v2-encoder-seed-10.pt --run_name action-predictor-supervised-v2-encoder-seed-10 --seed 10
```

Example usage for evaluation:
```bash
python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-supervised-v2-encoder-seed-10.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-supervised-v2-encoder-seed-10 --seed 10
```

### DINO

To train an action prediction model using DINO, you can run the script `train_action_prediction_v2.py`.

Example usage for fine-tuning (remove `--fine_tune` to train via linear probing):
```bash
python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-centering-seed-10.pt --run_name action-predictor-fine-tune-v2-encoder-centering-seed-10 --state_model gnn-state-model-v2-centering.pt --fine_tune --seed 10
```

Once the prediction model is fine-tuned (or linear probed), you can then evaluate it as follows:
```bash
python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-centering-seed-10.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-centering-seed-10 --seed 10
```

### GBT

To train an action prediction model using DINO, you can run the script `train_action_prediction_v2.py`.

Example usage for fine-tuning (remove `--fine_tune` to train via linear probing):
```bash
python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-barlow-twins-seed-10.pt --run_name action-predictor-fine-tune-v2-encoder-barlow-twins-seed-10 --state_model gnn-state-model-v2-barlow-twins.pt --fine_tune --seed 10
```

Once the prediction model is fine-tuned (or linear probed), you can then evaluate it as follows:
```bash
python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-barlow-twins-seed-10.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-barlow-twins-seed-10 --seed 10
```

### BGRL

To train an action prediction model using BGRL, you can run the script `train_action_prediction_bgrl_v2.py`.

Example usage for fine-tuning (remove `--fine_tune` to train via linear probing):
```bash
python train_action_prediction_bgrl_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-bgrl-seed-10.pt --run_name action-predictor-fine-tune-v2-encoder-bgrl-seed-10 --state_model gnn-state-model-v2-bgrl.pt --fine_tune --seed 10
```
Once the prediction model is fine-tuned (or linear probed), you can then evaluate it as follows:
```bash
python eval_action_prediction_bgrl_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-bgrl-seed-10.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-bgrl-seed-10 --seed 10
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for more details.

## Acknolwedgements 

I would like to give a special thanks to Santiago Ontanon for access to the CoG 2019 and CoG 2010 microRTS competition datasets!
