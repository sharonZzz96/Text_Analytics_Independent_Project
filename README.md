# Text-Analytics-Independent-Project

## Running the application 
### 1. Set up environment 
The `requirements.txt` file contains the packages required to run the model code. An environment can be set up after you cd to the repo path. 
#### With `virtualenv`
```bash
pip install virtualenv
virtualenv pennylane
source pennylane/bin/activate
pip install -r requirements.txt
python -m spacy download en
```

### 2. Run the application
Write one context in path_to_repo/data/context.txt and one question in path_to_repo/data/question.txt
 ```bash
python api.py
 ```
The predicted answer sentence will be printed in both terminal and saved in path_to_repo as a json file
