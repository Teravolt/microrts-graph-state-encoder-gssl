# Microrts Analyzer

Python code to parse replay data generated from the Real-Time Strategy AI testbed MicroRTS.

## Requirements

Please note that the replay parser has been tested on MacOS.
I believe this will work on Linux, but I am not 100% sure.
Below lists all Python packages for setting up the microrts replay parser.
You will need `Python 3.9`+ to use the replay parser.

- (REQUIRED) `numpy` (can be installed via `pip install numpy`)
- (REQUIRED) `tqdm` (can be installed via `pip install tqdm`)
- torch = "^2.0.1"
torchvision = "^0.15.2"
numpy = "^1.25.0"
pandas = "^2.0.2"
seaborn = "^0.12.2"
gradio = "^3.35.2"
jupyterlab = "^4.0.2"
tqdm = "^4.65.0"
- (OPTIONAL) `pylint` (Can be installved via `pip install pylint`) - This does not need to be installed unless you are contributing to the project.

## Environment Setup

We use `poetry` to set up our Python environment as it provides an easy and quick way to setup, update, and tear down python virtual environments. Please see the [Pipenv Documentation](https://pipenv-fork.readthedocs.io/en/latest/) for instructions on how to both install it on your preferred OS and use it.

If you prefer to use other methods to set up an environment, please make sure that the Python packages and version in the "Requirements" section are installed!

## Example Usage

To run the analyzer, you can run the following:
```
python main.py *input-directory*
```

To see all command line arguments, you can run the following:
```
python main.py --help
```

## Current limitations

- The replay parser only works over XML files. I am in the process of getting the parser to work over JSON replays.
- The replay parser only works on XML files with the above file structure. I will fix this to handle any file structure.
- The replay parser currently supports 2 player replays. I am in the process of getting the parser to work with 3+ player replays.

## Contributing 

There are many things I still have not done for the replay analyzer (Look at TODO for potential ideas). 

To contribute to the project: 
1. Fork the project
2. Create a feature branch 
3. Commit your changes
4. Push to the branch
5. Create a new pull request

## Versioning

Semantic Versioning (MAJOR.MINOR.PATCH)

## Authors

Pavan Kantharaju - Initial Work

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for more details.

## Acknolwedgements 

I would like to give a special thanks to Santiago Ontanon for the [microrts](https://github.com/santiontanon/microrts) testbed!