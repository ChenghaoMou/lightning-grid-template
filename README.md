# lightning-grid-template

This is a minimal project template for people use pytorch-lightning and grid.ai. It combines both lightning cli and grid config/actions so you don't need to pass on tons of parameters anywhere.

![GitHub](https://img.shields.io/github/license/ChenghaoMou/lightning-grid-template)

## Acknowledgements

-   [pytorch-lighting](https://www.pytorchlightning.ai/)
-   [grid.ai](https://www.grid.ai/)
-   [transformers](https://huggingface.co/transformers/)
-   [datasets](https://huggingface.co/docs/datasets/)
-   [poetry](https://python-poetry.org/)

## Features

-   `lightning.yaml` for model and dataset hyper-parameters
-   `grid.yaml` for grid experiment configs
-   Custom environment management with `poetry`

## Usage

1.  Create your own model and data modules like the ones defined in `run.py`
2.  Update `lightning.yaml` accordingly or create your own with lightning cli
3.  Update the environment with poetry to include any packages used in your project
4.  Update `grid.yaml` to include your credentials and tweak the hardware settings
5.  Let grid do the magic

## Todo

-   [ ] Overwrite lightning config hyper-parameters with grid, so we could do hyper-parameters search

## Disclaimer

This is **not** an official lightning project nor grid project.
Both lightning cli and grid actions/cli are in their early stages, so please do expect frequent changes.
