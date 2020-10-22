# Characted-Identification-in-Multi-Party-Dialogues

Main project repository for NLP course 2018.  
The goal of this project was to assign each mention in a dialogue to the entity it's referring to. This is also known as the coreference resolution problem.

## Requirements

Data is taken from https://github.com/emorynlp/character-mining

For pretrained word embeddings, GloVe is used.

Create a directory `pretrained_embeds/` in the same directory as this notebook.
Download twitter embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip
Unzip it and place file `glove.twitter.27B.25d.txt` in `pretrained_embeds/` directory.

Create an empty directory `data/` in the same directory as this notebook where all the processed data will get saved.

## Folders/Files discription

`json_data/` - Folder contains data downloaded from https://github.com/emorynlp/character-mining

`data/` - Folder contains the processed data that has been converted to features for training.

`models/` - trained models are saved here.

`feature_generation.ipynb` - Used to generate neural network feedable data(word embeddings).

`model.py` - Contains classes that defines neural network model.

`train.py` - Takes features and model, trains and saves models in models folder.

`evaluate.py` - Used to calculate the accuracy on test data by making use of saved models.
