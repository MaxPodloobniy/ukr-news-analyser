"""
Виконаємо такі частини аналізу тексту:
1) Аналіз частоти публікацій за годину:
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
import requests
import nltk
from nltk.tokenize import word_tokenize
from datetime import datetime
import matplotlib.pyplot as plt

# nltk.download('punkt_tab')


def download_stopwords_to_file(filename='ukrainian_stopwords.txt'):
    """Завантаження стоп-слів для української мови"""
    url = 'https://raw.githubusercontent.com/olegdubetcky/Ukrainian-Stopwords/main/ukrainian'
    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(r.text)
        print(f"Стоп-слова збережено у файл {filename}")
    except Exception as e:
        print(f"Помилка завантаження стоп-слів: {e}")


def preprocess_text(text):
    """Попередня обробка тексту, токенізація тексту, видалення стоп-слів і спеціальних символів"""
    # Видалення спеціальних символів
    text = re.sub(r'[^\w\s]', '', str(text))
    # Токенізація
    tokens = word_tokenize(text.lower())
    # Видалення стоп-слів
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


# ------------------------- Завантаження і попередня обробка тексту -------------------------
# Завантажуємо стоп-слова
with open('ukrainian_stopwords.txt', 'r', encoding='utf-8') as f:
    stopwords = set(line.strip() for line in f)

news_df = pd.read_csv('parsed_articles.csv')

news_df['processed_text'] = news_df['text'].apply(preprocess_text)

# Перетворюємо дату-час на datetime
news_df['date'] = pd.to_datetime(news_df['date'])



# ------------------------- 1) Аналіз частоти публікацій -------------------------
# Групування за годинами для аналізу частоти публікацій
news_df['hour'] = news_df['date'].dt.hour
hourly_counts = news_df.groupby('hour').size()

hourly_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Частота публікацій за годину')
plt.xlabel('Година')
plt.ylabel('Кількість публікацій')
plt.xticks(rotation=0)
plt.show()


# ------------------------- 2) Аналіз найчастіше вживаних слів (word cloud) -------------------------



