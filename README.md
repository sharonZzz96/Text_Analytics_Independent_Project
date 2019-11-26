# Text-Analytics-Independent-Project

## Author
Yueying (Sharon) Zhang


## Project
1. Topic: Question Answering

2. Task: The goal is to find the text for any new question and context provided. In this project, the answer to a question is always a part of the context and also a continuous span of context. So, the task is to find the sentence containing the right answer, in other words, answer sentence selection.

3. Dataset: Stanford Question Answering Dataset (SQuAD) v2.0
- Description: For each observation in the training set, we have a context, question, and text(answer). 

## Repo structure 
```
├── README.md                         <- You are here
│
├── api.py                            <- Main python file to run the application
│
├── models.py                         <- Python modula used in api.py
│
├── data                              <- Folder that contains user input data used in prediction
│
├── model                             <- Folder than contains trained model object
│   
├── notebooks                         <- Notebooks used in development including feature engineer and model selection
│
├── requirements.txt                  <- Python package dependencies 
```

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
Download infersent.pkl from https://drive.google.com/file/d/1anWf1G1WDYAvfSoce534dgurX6UQfN4C/view?usp=sharing and save the file as pat_to_repo/infersent.pkl

### 2. Run the application
Write one context in path_to_repo/data/context.txt and one question in path_to_repo/data/question.txt
 ```bash
python api.py
 ```
The predicted answer sentence will be both printed in terminal and saved in path_to_repo as a json file
