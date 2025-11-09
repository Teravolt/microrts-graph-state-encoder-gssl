# Code Repository for Paper "Pretraining Graph State Encoders for microRTS using Graph Self-Supervised Learning"

This repository contain Python scripts and Jupyter notebooks to (1) pretrain the graph state encoder used in the paper, and (2) run the cluster analysis and action prediction experiments contained in the paper.

## Installation

You will need Python 3.9+ to run the scripts and notebooks.

We use Poetry to set up our Python environment as it provides an easy and quick way to setup, update, and tear down python virtual environments.
Please see the [Poetry Documentation](https://python-poetry.org/) for instructions on how to both install it on your preferred OS and use it.

If you prefer to use other methods to set up a Python environment, please make sure that the Python packages and version from `pyproject.toml` are installed.
There is also a `requirements.txt` generated from `pyproject.toml` to help with installation outside of Poetry.

> Please note that this code was run on MacOS with Python 3.11.10.
This should work on Linux and other Python versions, but I am not 100% sure.
If you have any issues running this on other operating systems or other Python versions, let me know and I'll take a look!

## Training a Graph State Encoder

To pretrain a graph state encoder using Distillation with No Labels (DINO), you can run the script ` train_state_encoder_dino_v2_with_centering.py`.
The exact arguments we used in the paper are as follows:
```bash
python train_state_encoder_dino_v2_with_centering.py ../microrts-dataset/microrts-cog-2019-standard --read_from_zip --max_replay_length 64 --save_model gnn-state-model-v2-centering.pt --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --run_name pre-training-v2-encoder-centering
```

To pretrain a model using Graph Barlow Twins (GBT), you can run the script `train_state_encoder_barlow_twins_v2.py`.
The exact arguments we used in the paper are as follows:
```bash
python train_state_encoder_barlow_twins_v2.py ../microrts-dataset/microrts-cog-2019-standard --read_from_zip --max_replay_length 64 --save_model gnn-state-model-v2-barlow-twins.pt --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --run_name pre-training-v2-encoder-barlow-twins
```

To pretrain a model using Bootstraped Graph Latents (BGRL), you can run the script `train_state_encoder_bgrl_v2.py`.
The exact arguments we used in the paper are as follows:
```bash
python train_state_encoder_bgrl_v2.py ../microrts-dataset/microrts-cog-2019-standard --read_from_zip --max_replay_length 64 --save_model gnn-state-model-v2-bgrl.pt --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --run_name pre-training-v2-encoder-bgrl
```

The pretraining runs, exact parameters used in the paper, and pretrained models can be found in our Weights and Biases project: https://wandb.ai/pkthunder/microrts-graph-state-ssl?nw=nwuserpkthunder

## Action Prediction

To train an action prediction model using DINO, you can run the script `train_action_prediction_v2.py`.
The prediction runs, exact parameters used in the paper, and models can be found in our Weights and Biases project: https://wandb.ai/pkthunder/microrts-action-prediction?nw=nwuserpkthunder


Below is an example on fine-tuning an action prediction model using a graph state encoder pretrained by DINO:
```bash
python train_action_prediction_v2.py ../microrts-dataset/COG2019-competition/microrts-cog-2019-dataset-last-2-iteration --read_from_zip --max_replay_length 64 --device cpu --batch_size 16 --num_train_epochs 10 --frame_skip_freq 10 --frame_number_start 200 --save_model action-predictor-fine-tune-v2-encoder-centering-seed-10.pt --run_name action-predictor-fine-tune-v2-encoder-centering-seed-10 --state_model gnn-state-model-v2-centering.pt --fine_tune --seed 10
```

Once the prediction model is trained, you can then evaluate it as follows:
```bash
python eval_action_prediction_v2.py ../microrts-dataset/microrts-cog-2020-standard-last-two-iterations action-predictor-fine-tune-v2-encoder-centering-seed-10.pt --read_from_zip --max_replay_length 64 --device cpu --frame_skip_freq 10 --frame_number_start 200 --run_name eval-fine-tune-v2-encoder-centering-seed-10 --seed 10
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for more details.

## Acknolwedgements 

I would like to give a special thanks to Santiago Ontanon for the [microrts](https://github.com/santiontanon/microrts) testbed!