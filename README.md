# mlnd_capstone

## Required software
- python 2.7
- pip
- wget
- then ```pip install -r requirements.txt```

## Data files
these consist of:
- OHLC FX price data (`fxdata` folder)
- prepared datasets for training models (`input` folder)
- saved model output (`output` folder)

These can be downloaded from S3 storage using the following script:
`./download-data-files.sh`
Note: if running on Windows, then cygwin or similar unix-like environment is recommended. dos2unix conversion of this file may be needed on windows, depending on your environment and local git config.

## Code Overview 

### Jupyter Notebooks

- `CapstoneProjectReport.ipynb`  
  The document used to prepare the project report.
- `PrepareExtractsForReport.ipynb`  
  contains the code used to prepare figures and tables used in the report
- `PrepareDatasets.ipynb`  
  Used to prepare the datasets and save them to file.
- `TrainModels.ipynb`  
  Used to load datasets from file, create the models and running training sessions upon them
- `LoadAndEvaluateModel.ipynb`  
  Used to load a trained model from file, load a test dataset and evalute it's performance.
  
### Python Modules
- `datasets.py`  
   contains the `prepare_dataset` functions to generate the datasets. There are several variants of these, reflecting refinements as the project progressed.
- `metrics.py`  
   contains code to evaluate and report the performance of a model, given it's predictions and the true results.    
- `model01.py`  
   contains the implementation of the neural network model.
- `utils.py`  
   contains miscellaneous utility functions, including `load_1minute_fx_bars` for loading the raw CSV price data.
- `baseline01.py, baseline02.py`  
   contain the implemenation of the baseline models.

 
