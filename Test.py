#Here we will test our code 
from news_article import NewsArticle
from data_loader import dataLoader
class Test:
    #article = NewsArticle(title = "NASA confirms alien existence", text = "Test Article is being generated to see if the code runs or not. Have a nice day!", label = 0)
    #print(article.text)
    #print(article.title)
    #print(article) 
    #FIX ME : There is a formatting issue with the __repr__

    #Exception testing 
    #wrongArticle = NewsArticle(title = "", text = "", label = None)
    #print(wrongArticle)

    text1 = "Hel!!!lo !world!"
    text1 = dataLoader.cleanText(text1)
    print(text1)
