import pandas as pd
import csv
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import re
import gc
import requests
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.ldamodel import LdaModel
from gensim import corpora


# ------------------------- Функції обробки тексту -------------------------
def preprocess_text(text, nlp_preprocess_model):
    # Перевірка на NaN
    if isinstance(text, float):
        return []

    # Очистка тексту від спецсимволів
    text = re.sub(r'[^\w\s]', ' ', str(text))

    # Обробка тексту через spaCy
    doc = nlp_preprocess_model(text.lower())

    # Токенізація, лемантизація та видалення стоп-слів
    tokens = [
        token.lemma_  # лемантизація
        for token in doc  # токенізація
        if not token.is_stop  # видалення стоп-слів
           and not token.is_punct  # видалення пунктуації
           and not token.is_space  # видалення пробілів
           and token.lemma_.strip()  # перевірка на пусті токени
    ]
    gc.collect()

    return tokens  # Повертаємо список токенів


def extract_entities(text, nlp_ner_model):
    # Перевірка на NaN
    if isinstance(text, float):
        return []

    doc = nlp_ner_model(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "LOC", "GPE", "MISC"}]
    return entities


# Функція для лемматизації та обробки тексту
def preprocess_and_analyze(text, nlp_model, SIA_model):
    """
    Функція для лемантизації та аналізу тональності тексту використовуючи spaCy
    """
    # Перевірка на NaN
    if isinstance(text, float):
        return 0.0

    # Обробка тексту через spaCy
    doc = nlp_model(str(text).lower())

    # Токенізація, лемантизація та видалення стоп-слів
    processed_words = [
        token.lemma_
        for token in doc
        if not token.is_stop
           and not token.is_punct
           and not token.is_space
           and token.lemma_.strip()
    ]

    # Об'єднуємо оброблені слова назад у текст
    processed_text = ' '.join(processed_words)

    gc.collect()

    # Аналіз тональності
    return SIA_model.polarity_scores(processed_text)["compound"]


# ------------------------- 1) Аналіз частоти публікацій -------------------------
def freq_of_publication_analysis(news_df):
    # Групування за годинами для аналізу частоти публікацій
    news_df['hour'] = news_df['date'].dt.hour
    hourly_counts = news_df.groupby('hour').size()

    fig, ax = plt.subplots(figsize=(10, 6))
    hourly_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    plt.title('Частота публікацій за годину')
    plt.xlabel('Година')
    plt.ylabel('Кількість публікацій')
    ax.set_xticks(range(24))
    ax.tick_params(axis='x', rotation=0)
    plt.show()

    return fig



# ------------------------- 2) Аналіз найчастіше вживаних слів (word cloud) -------------------------
def words_freq_analysis(news_df):
    # Об'єднуємо всі оброблені тексти в один
    all_words = ' '.join([word for tokens in news_df['processed_text'] for word in tokens])

    # Параметри для хмари слів
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate(all_words)

    # Візуалізація
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    gc.collect()

    return fig


# ------------------------- 3) Тематичний аналіз (topic modeling) -------------------------
def find_news_themes(news_df):
    # Об'єднуємо токени назад у рядки для CountVectorizer
    news_df['processed_text_str'] = news_df['processed_text'].apply(' '.join)

    # Перетворення тексту в числовий формат (Bag of Words)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(news_df['processed_text_str'])

    # Перетворюємо дані у формат, придатний для gensim
    dictionary = corpora.Dictionary(news_df['processed_text'])  # тепер передаємо список списків токенів
    corpus = [dictionary.doc2bow(text) for text in news_df['processed_text']]

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=10,
        random_state=100,
        passes=10,
        alpha='auto'
    )

    # Виведення тем
    topics = lda_model.print_topics(num_words=5)
    print(topics)
    for idx, topic in topics:
        print(f"Тема {idx+1}: {topic}")

    gc.collect()

    return topics


# ------------------------- 4) Аналіз тональності за допомогою VADER -------------------------
def tonality_analysis_VADER(news_df, nlp_preprocess_model, load_new_dict=False):
    project_path = os.getcwd()
    tone_dict_path = os.path.join(project_path, 'tone_dict_uk.tsv')

    if load_new_dict:
        # Завантажуємо український словник тональності й зберігаємо у файлі в папці проєкту
        tone_dict_url = 'https://raw.githubusercontent.com/lang-uk/tone-dict-uk/master/tone-dict-uk.tsv'

        r = requests.get(tone_dict_url)
        with open(tone_dict_path, 'wb') as f:
            f.write(r.content)

    # Завантажуємо словник тональності та додаємо до VADER
    tone_dict = {}
    with open(tone_dict_path, 'r') as csv_file:
        for row in csv.reader(csv_file, delimiter='\t'):
            tone_dict[row[0]] = float(row[1])

    SIA = SentimentIntensityAnalyzer()
    SIA.lexicon.update(tone_dict)

    news_df['sentiment_score'] = news_df['title'].apply(lambda x: preprocess_and_analyze(x, nlp_preprocess_model, SIA))

    # Візуалізація: Гістограма тональності
    fig_1 = plt.figure(figsize=(10, 6))
    plt.hist(news_df['sentiment_score'], bins=40, color='skyblue', edgecolor='black')
    plt.title("Розподіл тональності новин")
    plt.xlabel("Тональність (compound score)")
    plt.ylabel("Кількість новин")
    plt.show()

    # Візуалізація: Середня тональність новин у часі
    news_df['date'] = pd.to_datetime(news_df['date'])
    sentiment_by_date = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].mean()

    fig_2 = plt.figure(figsize=(10, 6))
    sentiment_by_date.plot(marker='o', color='orange')
    plt.title("Середня тональність новин у часі")
    plt.xlabel("Дата")
    plt.ylabel("Середня тональність")
    plt.show()

    gc.collect()

    return fig_1, fig_2


# ------------------------- 5) Візуалізація згадок ключових осіб або подій за допомогою NER -------------------------
def extract_and_visualize_named_entities(news_df, nlp_ner_model):
    # Створюємо колонку з витягнутими NER для всього датафрейму одразу
    news_df['entities'] = news_df['processed_text_str'].apply(lambda x: extract_entities(x, nlp_ner_model))

    gc.collect()

    # Розгортаємо список сутностей в окремі рядки і підраховуємо
    entity_counts = news_df['entities'].explode().value_counts()

    # Конвертуємо в Counter для WordCloud
    entity_counts = Counter(dict(entity_counts))

    gc.collect()

    # Створюємо Word Cloud для згадок іменованих сутностей
    wordcloud = WordCloud(
        width=1000,
        height=600,
        background_color='white'
    ).generate_from_frequencies(entity_counts)

    # Відображаємо Word Cloud
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()

    return fig

