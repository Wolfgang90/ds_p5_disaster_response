# Disaster Response Pipeline Project

### Project summary

This project classifies messages sent during real disaster events. Labeled messages are extracted from .csv-files, preprocessed and uploaded to a database. To build the classifier data is extracted from the database, messages are tokenized and lemmatized and a Random Forest Classifier is trained on them. In a Flask-web app one can subsequently insert a message which is then classified by the classifier. Furthermore the distribution of genres of the provided messages for training and the categories of these messages can be displayed in the web app.

### File structure overview

```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py # File to preprocess data from .csv files in this folder and upload the result into the DisasterResponse-database
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py # File to define, train and save the classifier
|- classifier.pkl  # saved model 

- README.md
```

### How to run the project

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
