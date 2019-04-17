# Import libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing import sequence
import numpy as np  
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, accuracy_score, auc
# tensorflow is needed as a dependency for something else

'''
To run the following code, you can run the following 3 lines:

from model import review_invoices
review_invoices = review_invoices()
review_invoices.run()
'''

class review_invoices:
    '''
    I put all of David's code into a class. I split everything within the class
    into methods. I didn't adjust much of the code. Pretty much all I did was 
    combine the two csv's into one dataframe since the csv's had to be split
    into two.
    
    There is opportunity to adjust and restructure the methods within this class
    in a way that makes more sense. I just want to get a good framework for the
    code before we start expanding on it.
    '''
    
    def __init__(self):
        '''
        Initialize variables. Everything within this __init__ method gets
        run automatically when the review_invoices class gets called. This is
        a great place to define variables and possibly run methods automatically
        
        Every variable defined within this method can be called and viewed by 
        the user. Conversely, anything within the below methods is private.
        
        Remember that to make a variable callable/called by other methods, put 
        'self.' in front of the variable.
        '''
        # Load the data into dataframes
        self.df_1 = pd.read_csv('Data/data.csv')
        self.df_2 = pd.read_csv('Data/data2.csv')
        # Empty list for...
        self.documents = []
    
    def clean_df(self):
        # Combine the two dataframes into one dataframe
        df = self.df_1.append(self.df_2, ignore_index = True)
        # Fill null cells with the string 'blank'
        df = df.fillna('blank')
        # Parse out phone numbers into a new column, phone_num
        df['phone_num_1'] = df['Terms'].str.extract('(\(?\d\d\d\)?-? ?\.?\d\d\d-?\.? ?\d\d\d\d?)')
        # Remove the phone numbers from the Terms column
        df['Terms'] = df['Terms'].replace('(\(?\d\d\d\)?-? ?\.?\d\d\d-?\.? ?\d\d\d\d?)',
            '', regex = True)
        # Extract email addresses
        df['email'] = df['Terms'].str.extract('(\S+@\S+)')
        # Remove email addresses
        df['Terms'] = df['Terms'].replace('(\S+@\S+)', '', regex=True)
        # Convert dataframe columns to series
        self.X = df["Terms"]
        self.y = df["Chargeback"]

    def link_words(self):
        # Further clean and then lemmatize
        # Some of the below could be moved to clean_df()
        stemmer = WordNetLemmatizer()
        for sen in range(0, len(self.X)):
            document = re.sub(r'\W', ' ', str(self.X[sen]))
            document = document.lower()
            document = document.split()
            document = [stemmer.lemmatize(word) for word in document]
            document = ' '.join(document)
            self.documents.append(document)

    def vectorize(self):
        from sklearn.feature_extraction.text import TfidfVectorizer  
        tfidfconverter = TfidfVectorizer(
            max_features=2000,
            min_df=10,
            max_df=0.7,
            stop_words=stopwords.words('english'))  
        self.X = tfidfconverter.fit_transform(self.documents).toarray()  

    def partition(self):
        from sklearn.model_selection import train_test_split  
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, 
            self.y, test_size=.2, random_state=1)  

    def model(self):
        model = Sequential()
        model.add(Dense(2000, input_dim = 2000, activation = 'relu'))
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dense(500, activation = 'relu'))
        model.add(Dense(1, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
            metrics = ['accuracy'])
        return model

    def run(self):
        '''
        This method can be called as an easy way to run all of the above methods
        and the commands to get output. 
        
        The other easy alternative is to include all of this stuff in the
        __init__ method so it get's run automatically when the class is called.
        Splitting all of this into its own method just makes the class easier to
        control
        '''
        self.clean_df()
        self.link_words()
        self.vectorize()
        self.partition()
        self.model()
        
        # -------- call model -------- 
        model = self.model()
        
        # -------- fit  -------- 
        model.fit(self.X_train,self.y_train, epochs = 10, batch_size = 512, 
            verbose = True)
        
        # -------- predict  -------- 
        pred = model.predict_classes(self.X_test)
        
        # -------- Confusion Matrix -------- 
        matrix = pd.DataFrame(confusion_matrix(self.y_test,pred, 
            labels = [x for x in range(0,2)]))
        matrix
        
        # -------- accuracy -------- 
        accuracy_score(self.y_test,pred)
        
        # -------- summary -------- 
        model.summary()
