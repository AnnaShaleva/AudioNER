import os

### PATHS ###

# Main Paths
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
CATEGORIES_PATH = os.path.join(PROJECT_ROOT_PATH, "categories/")
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data/")


### PARAMETRS ###

## ElasticSearch Settings
ELASTIC_HOST = "168.61.86.72:8120"
ELASTIC_TIMEOUT = 6000

## Spectrogram settings
FREQUENCY_TRESHHOLD = 1000
NEIGHBORHOOD_SIZE = 12
AMPLITUDE_DIFFERENCE_TRASHHOLD = 100


## MFCC parametrs

# Number of MFCC features
N_INPUT = 26
N_CONTEXT = 9

## Model parametrs
N_CLASSES = 2
N_SUBCLASSES = 10
VALIDATION_SPLIT = 0.15

## Data format
HEIGHT = 20
LENGTH = 494

