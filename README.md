# Topic-modeling

This package contains script, code files and tools to compute labels for topics Modelling using LDA, Doc2vec and Word2vec (over phrases) models.

# Pre-Trained Models

Doc2Vec trained model
Word2vec trained model(ngrams)

# Additional support files

preInstalled JSON formatted texts.
Wordvec Phrase list
Filtered/Short Document titles

# Some Python Library Requirements

Gensim
Numpy
Pandas
LDA

# Running the System

  Directly running the pre-trained system and get the labels.(Training the System not needed for this step)

  Download the trnscript files from S3 bucket and place it at local folder.

  Make sure your data topic file is in .tar format.

  Run get_data.py so that the data will converts from .tar in to JSON format and creates a JSON file.
  
  Then execute tokens.py to extract tar file, read transcripts data and generate tokens.


# Input Format

The input format if just need to run the model

Topic file: One line per topic (displaying top-N words) in .csv format. Path to this file be updated in token.py

The input format for training a new model.

Transcripts backup from MongoDB and place it at Amazon S3 bucket.

# Directory Structure and Files

There are 2 main directories. First is is model_run. It has files which are used to directly run and give us tokens, The second directory is training.The training directory contains script and code for training the embedding models. 


  model_run/s3.py: This script download the data from S3 Bucket and stores in to local folder.
  
  model_run/token.py: This script extract the data which is in tar format in to JSON format and stores in to another location. Also read transcripts data and generate tokens
  
  model_run/filter_data.py: This script filters the duplicate data and cleans the unnecessary text like duplicate words.
  
  model_run/nlp.py: This scripts extracts the Ngrams and Bigrams. And then extracts the tokens from the text.
  
   model_run/toy_data/Vizulation.py: This script dipicts the visualisation chart based upon corpus and dictionaries from the text. It has CV plot
  
  model_run/pre_trained_models. Directory to place trained doc2vec and word2vec models.
  
  model_run/data : Place your topic data file here.
  
  training/generate_model_lda: This scripts generates the LDA model based on KL divergence.

  
  # Datasets
  
  A Topic file given which has > 100000 topics with its 10 terms from four domains namely experts meetings, expert calls, expert suggestion
  Annotated files: For each topic 19 labels were annotated.



