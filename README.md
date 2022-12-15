
# Udacity Data Scientist Homework Assignment 2
This contains the code for homework 2 in the Udacity Data Scientist Nano Degree. Based on data provided by Figure Eight a model was trained to classify disaster messages into disaster categories. The model was incorporated into a web app using Flask that allows users to classify messages. Furthermore, additional insides into the training data was porvided through the web app using visualizations made with the plotly library.

# Libraries used
**sys**: provides access to system-specific parameters and functions

**json**: provides an API to encode and decode data in JSON format

**plotly**: provides interactive and publication-quality graphs

**pandas**: provides high-level data structures and functions for data analysis

**numpy**: provides scientific computing tools

**pickle**: provides an object persistence API

**nltk**: provides tools for natural language processing

**flask**: provides a web framework for applications

**plotly**: provides tools for creating interactive visualizations

**sklearn**: provides machine learning tools

**sqlalchemy**: provides a SQL toolkit and ORM for databases

# Files
* \app
*   run.py: flask code to run web app
*   \templates
*     master.html: web app main page
*     go.html: web app results page
* \data
*   disaster_categories.csv: disaster categories dataset
*   disaster_messages.csv: disaster messages dataset
*   DisasterResponse.db: disaster response data in database (output of process_data.py)
*   process_data.py: ETL process
* \models
*   classifier.pkl: exportet disaster message classification model (result of train_classifier.py)
*   train_classifier.py: ML pipeline

# Project Components
1. **ETL Pipeline**
`/data/process_data.py` contains a data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database

2. **ML Pipeline**
`/models/train_classifier.py` contains a machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

3. **Flask Web App**
`/app/run.py` contains the web app that lets users classify their messages based on the created ML Pipeline and shows visualizations to give further insides into the dataset used for training the model.


# Instructions
To launch the web app follow these steps:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the project's root directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/ to open the web app


# Licensing, Authors and Acknowledgements
Dataset source:
https://appen.com/datasets/combined-disaster-response-data/?%2338%3B

Dataset credit:
Appen (formerly figure 8)
