# TAED2_SmartHealth.AI

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

SmartHealth-AI is a project developed for the TAED2 course at UPC, designed to put MLOps concepts into practice. It combines a Random Forest model that predicts a user’s obesity level from health and lifestyle data with Google gemini that provides personalized health advice. Everything runs through a FastAPI backend, making it easy to integrate into a web application.

The API is available at http://10.4.41.72:8000/ for UPC members until the end of the winter semester.

## Project Organization

```
TAED2_SmartHealth-AI/
├── LICENSE            <- Open-source license
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project
├── pyproject.toml     <- Project configuration with package metadata
├── setup.cfg          <- Configuration file for flake8
├── params.yaml        <- Model parameters configuration
├── dvc.yaml           <- DVC pipeline configuration
├── dvc.lock           <- DVC lock file
├── uv.lock            <- UV dependency lock file
├── .gitignore         <- Git ignore rules
│
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   ├── raw            <- The original, immutable data dump
│   ├── __init__.py
│   └── train_obesity.py  <- Scripts to download or generate data
│
├── taed2_smarthealth_ai   <- Source code for use in this project
│   ├── __pycache__
│   ├── __init__.py
│   ├── api                <- API implementation for model serving
│   │   ├── api.py         <- API endpoints
│   │   └── static         <- Static files for web interface
│   │       └── index.html <- Web UI
│   └── data               <- Data-related utilities
│       ├── __init__.py
│       └── train_obesity.py  <- Script to train the obesity classification model
│
├── tests              <- Unit and integration tests
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models
│   └── modelcard.md   <- Model card documentation
│
├── notebooks          <- Jupyter notebooks
│   └── references     <- Data dictionaries and explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   ├── emissions.csv  <- Carbon emissions report
│   ├── Laboratory_1_report.pdf <- Milestone 1 and 2 report
│   ├── lint_report.txt
│   └── metrics.json
│
├── gx                 <- Great Expectations folder
    ├── checkpoints/
    ├── data_docs/
    ├── expectations/
    ├── plugins/
    ├── validation_definitions/
    └── great_expectations.yml
