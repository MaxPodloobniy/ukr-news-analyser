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

from tools import *
from report_generator import generate_markdown_report
import spacy



def main():
    # ------------------------- Завантаження та формування моделей -------------------------

    # Пайплайн для попередньої обробки (з лемантизацією)
    try:
        nlp_preprocess = spacy.load("uk_core_news_lg")
    except OSError:
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "uk_core_news_lg"])
        nlp_preprocess = spacy.load("uk_core_news_lg")

    # Відключаємо ВСЕ окрім токенізатора та лематизатора
    pipes_to_disable = []
    for name, _ in nlp_preprocess.pipeline:
        if name not in ['tok2vec', 'lemmatizer']:  # Потрібні для лематизації
            pipes_to_disable.append(name)
    print(f"Відключаємо пайпи: {pipes_to_disable}")
    nlp_preprocess.disable_pipes(*pipes_to_disable)

    # Завантажуємо стоп-слова
    with open('/kaggle/input/ukrainian-stoop-words/ukrainian_stopwords.txt', 'r', encoding='utf-8') as f:
        custom_stop_words = set(line.strip() for line in f)
        # Додаємо кастомні стоп-слова до стандартних стоп-слів spaCy
        for word in custom_stop_words:
            nlp_preprocess.vocab[word].is_stop = True

    # Пайплайн для NER
    nlp_ner = spacy.load("uk_core_news_lg")
    nlp_ner.disable_pipes(["tok2vec", "morphologizer", "lemmatizer", "attribute_ruler"])


    # ------------------------- Обробка даних -------------------------
    dtypes = {
        'title': 'string',
        'url': 'string',
        'date': 'string',
        'text': 'string'
    }

    news_df = pd.read_csv('/kaggle/input/parsed-ukrainian-news/parsed_articles.csv', dtype=dtypes)
    news_df['date'] = pd.to_datetime(news_df['date'])
    print('Dataset loaded')

    news_df['processed_title'] = news_df['title'].apply(lambda x: preprocess_text(x, nlp_preprocess))
    print('Titles processed')

    news_df['processed_text'] = news_df['text'].apply(lambda x: preprocess_text(x, nlp_preprocess))
    print('Texts processed')

    news_df = news_df.dropna().reset_index(drop=True)

    gc.collect()

    # ------------------------- Аналіз даних -------------------------

    # Аналіз частоти публікацій новин
    publication_freq_figure = freq_of_publication_analysis(news_df)

    # Аналіз найчастіше вживаних слів (word cloud)
    word_freq_cloud = words_freq_analysis(news_df)

    # Аналіз тональності(VADER)
    tonality_hist, average_tonality_over_time = tonality_analysis_VADER(news_df, nlp_preprocess)

    # Аналіз та візуалізація ключових осіб та подій
    named_ent_freq_cloud = extract_and_visualize_named_entities(news_df, nlp_ner)

    # Виявлення перепакованих (копіпаст) новин
    copypast_freq_hist, copypast_examples = detect_repackaged_news(news_df)

    # Виявлення клікбейтних заголовків
    cosine_similarity_freq, click_bait_text = analyze_title_text_similarity(news_df)

    # Виявлення маніпулятивності в новинах
    man_part_freq, man_part_text = analyze_manipulative_language(news_df, "/kaggle/input/ukrainian-manipulation-words/Manipulation words.txt")

    figures = {
        'publication_freq': publication_freq_figure,
        'wordcloud': word_freq_cloud,
        'all_tonality': tonality_hist,
        'tonality_per_time': average_tonality_over_time,
        'ner_visualization': named_ent_freq_cloud,
        'repackaged_news': copypast_freq_hist,
        'title_text_similarity': cosine_similarity_freq,
        'manipulative_language': man_part_freq
    }

    text_results = {
        'repackaged_news': copypast_examples,
        'title_text_similarity': click_bait_text,
        'manipulative_language': man_part_text
    }

    generate_markdown_report(news_df, figures, text_results)



if __name__ == "__main__":
    main()
