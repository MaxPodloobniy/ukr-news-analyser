"""
Виконаємо такі частини аналізу тексту:
1) Аналіз частоти публікацій за годину або день:
2) Аналіз найчастіше вживаних слів (word cloud)
3) Тематичний аналіз (topic modeling)
4) Семантичний аналіз за допомогою Word2Vec або BERT
5) Класифікація статей за категоріями
6) Візуалізація згадок ключових осіб або подій за допомогою NER
Так як в nltk нема української мови для видалення стоп-слів будемо користуватись цим словником:
'https://raw.githubusercontent.com/olegdubetcky/Ukrainian-Stopwords/main/ukrainian'
Для аналізу тональності будемо використовувати VADER з nltk і pymorphy2 для морфологічного аналізу так як в nltk нема
української мови:
https://github.com/kmike/pymorphy2.git
"""



import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime



def preprocess_text(text):
    """Попередня обробка тексту"""
    # Видалення спеціальних символів
    text = re.sub(r'[^\w\s]', '', str(text))
    # Токенізація
    tokens = word_tokenize(text.lower())
    # Видалення стоп-слів
    tokens = [t for t in tokens if t not in stopwords]
    return tokens



stop_words = set(stopwords.words('ukrainian') + stopwords.words('english'))

news_df = pd.read_csv('parsed_articles.csv')

news_df['text'] = preprocess_text(news_df['text'])

print(news_df['text'])

