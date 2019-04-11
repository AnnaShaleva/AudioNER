import os

### PATHS ###

# Main Paths
PROJECT_ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
CATEGORIES_PATH = os.path.join(PROJECT_ROOT_PATH, "categories/")
DATA_PATH = os.path.join(PROJECT_ROOT_PATH, "data/")


### PARAMETRS ###

# ElasticSearch Settings
ELASTIC_HOST = "localhost:9200"
