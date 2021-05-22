# lightning-grid-template
A minimal template for [pytorch-lightning](https://www.pytorchlightning.ai/) and [grid.ai](https://www.grid.ai/)

## Structure

- `run.py` contains a minimal tweet classification model and dataset with Huggingface's [transformers](https://huggingface.co/transformers/) & [datasets](https://huggingface.co/docs/datasets/)
- `lightning.yaml` includes all the hyper-parameters needed for both the model and the data specified in `run.py`
- `grid.yaml` has all the training specification for a grid experiment, including a hack to set up your own environment

## Usage
- Update the credentials in the `grid.yaml` to use your own;
- Update your model and dataset in `run.py` and `lightning.yaml` accordingly;
- Update your environment with `poetry`
    - `pip install poetry`
    - `poetry add package` instead of  `pip install package`

## Todo
- [ ] Overwrite lightning config hyper-parameters with grid, so we could do hyper-parameters search

## Disclaimer

This is **not** an official lightning project nor grid project.
Both lightning cli and grid actions/cli are in their early stages, so please do expect frequent changes.
