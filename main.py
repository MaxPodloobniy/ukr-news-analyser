"""
Виконаємо такі частини аналізу тексту:
1) Аналіз частоти публікацій за годину:
2) Аналіз найчастіше вживаних слів (word cloud)
3) Тематичний аналіз (topic modeling)(LDA-- вийшла якась фігня)
4) Аналіз тональності за допомогою VADER
5) Візуалізація згадок ключових осіб або подій за допомогою NER
Так як в nltk нема української мови для видалення стоп-слів будемо користуватись цим словником:
'https://raw.githubusercontent.com/olegdubetcky/Ukrainian-Stopwords/main/ukrainian'
Для аналізу тональності будемо використовувати VADER з nltk і pymorphy2 для морфологічного аналізу так як в nltk нема
української мови:
https://github.com/kmike/pymorphy2.git
"""

import spacy
from tools import *

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
    news_df = pd.read_csv('parsed_articles.csv')

    news_df['processed_text'] = news_df['text'].apply(lambda x: preprocess_text(x, nlp_preprocess))
    news_df['date'] = pd.to_datetime(news_df['date'])


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


if __name__ == "__main__":
    main()
