"""PURPOSE OF THIS CLASS
    This class is like a data model for our domain. 
    Each instance of this object will hold the article's text content and the label of it being real or fake for binary classification. 
    Using this class we can pass around articles in our pipeline like from DataLoader class to PreProcessor in a structures way rather than juggling up seperate pieces of information."""

#Variables Used
"""
title (String) : Holds the Article title
text (String) : Holds the news article
label (int) : holds 0 for FAKE or 1 for REAL"""

from myExceptions import DataFormatException
#Constructor
class NewsArticle:
    def __init__(self, title: str, text: str, label: int):
        if title is None or text.strip() == "" or label is None:
            raise DataFormatException()
        
        self.title = title
        self.text = text
        self.label = label


#Convenience methods 

# NOTE: if i use this def __str__(self), then i can access it by print(obj) but if i use __repr__ then i can directly print obj

#Method for human readable representation
def __repr__(self):
    #it starts with 'f' to indicate formatted string literal, [:30] ensures that only 30 letters are displayed in the console to avoid overcrowding and !r ensures that string is displayed with quotes and escapes
    return f"NewsArticle(label = {self.label}, titleSnippet = {self.title[:30]!r}...)"

#In python we dont need getters and setters as we can directly access the variables 
