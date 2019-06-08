"""Model to predict work order chargebacks for Invitation Homes

"""

# Import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence
from keras.utils.vis_utils import plot_model
from IPython.display import clear_output
from IPython.display import display
import time, sys
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (confusion_matrix, accuracy_score, auc, 
    balanced_accuracy_score, precision_score, average_precision_score,
    recall_score, classification_report)
# tensorflow is needed as a dependency

class review_invoices:
    '''
    First explore, clean, and normalize the data. Next train a neural network
    using Keras to predict who is liable for a work order. Finally print 
    output of the model.
    '''

    def __init__(self):
        
        """
        Import the data which should be comprised of three columns:
            1) WO #
            2) Chargeback
            3) Terms
        
        Create a Pandas dataframe of the data and make it a class variable.
        """
        
        print('\nInitializing...')
        # Load the data into dataframes
        self.df_1 = pd.read_csv('Data/data.csv')
        self.df_2 = pd.read_csv('Data/data2.csv')
        df = self.df_1.append(self.df_2, ignore_index = True)
        # Rename column headers
        df.rename(columns = {'WO #':'work_order_id', 'Chargeback':'liability', 
            'Terms':'work_order'}, inplace = True)
        self.df = df
        # Update Pandas settings. View full contents of each column
        pd.set_option('display.max_colwidth', -1)
        # Display up to 10 columns
        pd.set_option('display.max_columns', 10)
        # A check for null values
        self.null = df.isnull().values.any()

    def explore_data(self):
        
        "View some basic details regarding the data"
        
        print('\nExploring the data...')
        # Define the raw dataframe
        df = self.df
        # Print basic info about dataframe
        print('Original dataframe info:')
        print('-' * 40)
        df.info()
        print('-' * 40)
        # Print out first 5 rows of the df
        print(f'Are there any null values? {self.null}')
        print('Printing the first 5 rows of the original dataframe:')
        display(df.head())
        # Create csv of duplicate terms to be audited
        duplicate_terms = df[df.duplicated(subset=['work_order'], keep = False)]
        self.duplicate_terms = duplicate_terms.sort_values(by=['work_order'])
        duplicate_terms['work_order_id'].nunique()
        # Create csv of duplicate work order numbers to be audited
        duplicate_wo = df[df.duplicated(subset=['work_order_id'], keep = False)]
        self.duplicate_wo = duplicate_wo.sort_values(by=['work_order_id'])

    def clean_df(self):
        
        "Clean the column containing work order text"
        
        print('\nCleaning the dataframe...')
        df = self.df
        # Remove any rows with a null cell
        if self.null is True:
            df = df.dropna()
        # Remove rows with invalid terms
        print('Dropping work orders with invalid text: "#NAME?"')
        df = df.drop(df[df['work_order'] == '#NAME?'].index)
        # Parse out phone numbers into a new column, phone_num
        print('Extracting and removing phone numbers')
        df['phone_num'] = df['work_order'].str.extract(
            '(\(?\d\d\d\)?-? ?\.?\d\d\d-?\.? ?\d\d\d\d?)')
        # Remove the phone numbers from the work_order column
        df['work_order'] = df['work_order'].replace(
            '(\(?\d\d\d\)?-? ?\.?\d\d\d-?\.? ?\d\d\d\d?)', '', regex = True)
        print('Extracting and removing email addresses')
        # Extract email addresses and put into separate column
        df['email'] = df['work_order'].str.extract('(\S+@\S+)')
        # Remove email addresses from work_order column
        df['work_order'] = df['work_order'].replace('(\S+@\S+)', '', regex = True)
        print('Removing some meaningless words from work order templates')
        # Remove "Contact:", "Email:", "Phone:" from each work order
        df['work_order'] = df['work_order'].replace('(Contact:|Email:|Phone:)', 
            '', regex=True)
        print('Extracting and removing property ID\'s')
        # Extract the property ID from the end of each work order
        df['property_id'] = df['work_order'].str.rsplit(' ', 1).str[1]
        # Remove the property ID from each work order
        df['work_order'] = df['work_order'].str.rsplit(' ', 1).str[0]
        # Replace any non-word characters from work_order column with a space
        print('Replacing all non-word characters with a space')
        df['work_order'] = df['work_order'].str.replace('\W', ' ', regex = True)
        # Make the work_order column all lower case
        print('Making work_order column all lower case')
        df['work_order'] = df['work_order'].str.lower()
        print('Turning column of strings into column of lists (This takes some '
            'time)')
        df['work_order'] = df['work_order'].apply(word_tokenize)
        # Make clean dataframe callable outside of the method
        # The index was messed up after removing some rows, need to reset_index
        df = df.reset_index(drop = True)
        self.df_clean = df
        # Review some of the changes made to the data
        df_clean = df
        print('\nCleaned dataframe info:')
        print('-' * 40)
        print(df_clean.info())
        print('-' * 40)
        print('\nPrinting the first 5 rows of the clean dataframe')
        display(df_clean.head())
        # Convert dataframe columns to series for later method use
        self.X = df["work_order"]
        self.y = df["liability"]

    def link_words(self):
        
        "Lemmatize the work order column"
        
        print('\nLemmatizing... This one takes some time too.')
        # Define the work_order column as X
        X = self.X
        # Create an empty list called documents used to append lemmatized text
        documents = []
        stemmer = WordNetLemmatizer()
        # Lemmatize each word from each list of words, one at at time
        # Join those words together into strings, like they started
        # Append each string onto the documents list
        for sen in range(0, len(X)):
            document = X[sen]
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            documents.append(document)
        self.documents = documents
        # Print out first five items in documents list
        print('We\'ve turned the work_order column into a list called '
            '"documents"')

    def vectorize(self):
        
        "Vectorize the work order column"
        
        print('\nVectorizing...')
        tfidfconverter = TfidfVectorizer(
            max_features=2000,
            min_df=10,
            max_df=0.7,
            stop_words=stopwords.words('english')
        )
        self.X = tfidfconverter.fit_transform(self.documents).toarray()  

    def partition(self):
        
        "Split data into training and test groups"
        
        print('Splitting the data into training and test groups...')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
            self.y, test_size=.2, random_state=1)  

    def model(self):
        
        "Train the neural network to predict chargebacks"
        
        print('\nSetting up the model...')
        model = Sequential()
        model.add(Dense(100, input_dim = 2000, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        print('Compiling the model...')
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
        print('Fitting the model...')
        model.fit(self.X_train, self.y_train, epochs = 10, batch_size = 128, 
            verbose = True)
        return model

    def visualize(self):
        
        "Print output regarding the trained model"
        
        print('\nVisualizing the model output...')
        model = ri.model()
        pred = model.predict_classes(self.X_test)
        matrix = pd.DataFrame(confusion_matrix(self.y_test, pred, 
            labels = [x for x in range(0,2)]))
        print(f'Confusion matrix:\n {matrix}')
        print(f'''
        Accuracy score: {accuracy_score(self.y_test, pred)}
        Balanced accuracy score: {balanced_accuracy_score(self.y_test, pred)}
        Precision score: {precision_score(self.y_test, pred)}
        Average precision score: {average_precision_score(self.y_test, pred)}
        Recall score: {recall_score(self.y_test, pred)}
        Classification report: {classification_report(self.y_test, pred)}
        ''')
        print(model.summary())

# Making this file executable
# can enter "python3 model.py" in terminal and the full model will run
if __name__ == "__main__":
    ri = review_invoices()
    ri.explore_data()
    ri.clean_df()
    ri.link_words()
    ri.vectorize()
    ri.partition()
    ri.visualize()
