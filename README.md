# Capstone
MSDS 498 Capstone
Spring 2019

This is the repository where all code related to the Northwestern MSDS 498 Capstone class will be shared.

## Running code

Working from the directory where the model.py file is saved...  
Assuming you have the two data files in a directory called Data...


from model import review_invoices  
review_invoices = review_invoices()  
review_invoices.clean_df()   # run the clean_df() method  
review_invoices.run()   # run the run() method, which runs everything
