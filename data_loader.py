# data_loader.py (updated)
import re
import pandas as pd
import os
from myExceptions import DataFormatException
from news_article import NewsArticle

class DataLoader:

    @staticmethod
    def clean_text(text):
        text = str(text)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return cleaned_text

    @staticmethod
    def find_data_files():
        """Search for potential data files in the current directory"""
        data_files = []
        for file in os.listdir('.'):
            if file.endswith('.csv') or file.endswith('.xlsx') or file.endswith('.xls'):
                data_files.append(file)
        return data_files

    @staticmethod
    def load_data(file_path=None):
        articleList = []
        
        # If no file specified, try to find one
        if file_path is None:
            data_files = DataLoader.find_data_files()
            if not data_files:
                print("Error: No data files found in directory.")
                return articleList
            file_path = data_files[0]  # Use the first found file
            print(f"Using data file: {file_path}")
        
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
                return articleList
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return articleList

        # Print column names and first few rows for debugging
        print(f"Columns in dataset: {list(data.columns)}")
        print("First few rows of data:")
        print(data.head())
        
        # Check if the file has the correct column headers
        columns = data.columns
        columns_lower = [col.lower() for col in columns]

        # Try to find the correct columns if they're not in the expected order
        title_col = None
        text_col = None
        label_col = None
        
        for col in columns:
            if 'title' in col.lower():
                title_col = col
            elif 'text' in col.lower() or 'content' in col.lower():
                text_col = col
            elif 'label' in col.lower() or 'class' in col.lower() or 'category' in col.lower():
                label_col = col
        
        if not (title_col and text_col and label_col):
            raise DataFormatException("Could not find required columns: title, text, label")
        
        for index, row in data.iterrows():
            title = row[title_col]
            text = row[text_col]
            label = row[label_col]
            
            # Handle NaN values
            if pd.isna(title) or pd.isna(text) or pd.isna(label):
                continue
                
            # Print the types for debugging
            print(f"Row {index}: title type={type(title)}, text type={type(text)}, label type={type(label)}")
            print(f"Label value: {label}")
                
            try:
                article = NewsArticle(title, text, label)
                articleList.append(article)
            except DataFormatException as e:
                print(f"Error creating NewsArticle for row {index}: {e}")
                print(f"Title: {title}")
                print(f"Text: {text}")
                print(f"Label: {label}")
                continue

        print(f"Loaded {len(articleList)} articles from {file_path}")
        return articleList