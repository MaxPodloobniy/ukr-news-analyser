"""
Виконаємо такі частини аналізу тексту:
1) Аналіз частоти публікацій за годину:
2) Аналіз найчастіше вживаних слів (word cloud)
3) Тематичний аналіз (topic modeling)
4) Аналіз тональності за допомогою VADER
5) Візуалізація згадок ключових осіб або подій за допомогою NER
Крім стоп-слів з spacy додамо також свій словник стоп-слів
Для аналізу тональності будемо використовувати VADER і pymorphy2 для морфологічного аналізу
української мови:
https://github.com/kmike/pymorphy2.git
"""

import spacy
from tools import *
from report_generator import generate_analytics_report

# nltk.download('punkt_tab')
# nltk.download('vader_lexicon')


def main():
    # ------------------------- Завантаження та формування моделей -------------------------

    # Пайплайн для попередньої обробки (з лемантизацією)
    nlp_preprocess = spacy.load("uk_core_news_lg")
    nlp_preprocess.disable_pipes(["ner", "parser"])

    # Пайплайн для NER
    nlp_ner = spacy.load("uk_core_news_lg")
    nlp_ner.disable_pipes(["tok2vec", "morphologizer", "lemmatizer", "attribute_ruler"])

    # Завантажуємо стоп-слова
    with open('ukrainian_stopwords.txt', 'r', encoding='utf-8') as f:
        custom_stop_words = set(line.strip() for line in f)
        # Додаємо кастомні стоп-слова до стандартних стоп-слів spaCy
        for word in custom_stop_words:
            nlp_preprocess.vocab[word].is_stop = True


    # ------------------------- Обробка даних -------------------------
    dtypes = {
        'title': 'string',
        'url': 'string',
        'date': 'string',
        'text': 'string'
    }

    news_df = pd.read_csv('parsed_articles.csv', dtype=dtypes)

    news_df['processed_text'] = news_df['text'].apply(lambda x: preprocess_text(x, nlp_preprocess))
    news_df['date'] = pd.to_datetime(news_df['date'])

    gc.collect()


    # ------------------------- Аналіз даних -------------------------

    # Аналіз частоти публікацій новин
    publication_freq_figure = freq_of_publication_analysis(news_df)

    # Аналіз найчастіше вживаних слів (word cloud)
    word_freq_cloud = words_freq_analysis(news_df)

    # Аналіз основних тем (topic modeling)
    news_topics = find_news_themes(news_df)

    # Аналіз тональності(VADER)
    tonality_hist, average_tonality_over_time = tonality_analysis_VADER(news_df, nlp_preprocess)

    # Аналіз та візуалізація ключових осіб та подій
    named_ent_freq_cloud = extract_and_visualize_named_entities(news_df, nlp_ner)

    figures = {
            'publication_freq': publication_freq_figure,
            'wordcloud': word_freq_cloud,
            'all_tonality': tonality_hist,
            'tonality_per_time': average_tonality_over_time,
            'ner_visualization': named_ent_freq_cloud
        }

    text_results = {
        'formed_topics': news_topics
    }

    report_file = generate_analytics_report(news_df, figures, text_results)


if __name__ == "__main__":
    main()
