import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
import re
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------- Функції обробки тексту -------------------------
def preprocess_text(text, nlp_model):
    if isinstance(text, float):
        return []

    # Попередня очистка та обрізання
    text = str(text)
    if len(text) > 999999:
        text = text[:999999]

    text = re.sub(r'[^\w\s]', ' ', text).lower().strip()

    if not text:
        return []

    try:
        doc = nlp_model(text)
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
               and not token.is_punct
               and not token.is_space
               and len(token.lemma_.strip()) > 1
        ]
        return tokens
    except Exception as e:
        print(f"Помилка обробки тексту довжиною {len(text)}: {e}")
        return []


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


# ------------------------- 3) Аналіз тональності за допомогою VADER -------------------------
def analyze_sentiment_from_tokens(tokens, SIA_model):
    # Просто об'єднуємо токени в текст для VADER
    processed_text = ' '.join(tokens)
    return SIA_model.polarity_scores(processed_text)["compound"]


def tonality_analysis_VADER(news_df, load_new_dict=False):
    tone_dict_path = '/kaggle/input/ukrainian-tone-dictionary/tone_dict_uk.tsv'

    # Оптимізація 1: читаємо словник одразу в dict
    tone_dict = pd.read_csv(tone_dict_path, delimiter='\t', header=None,
                            names=['word', 'score']).set_index('word')['score'].to_dict()

    # Оптимізація 2: створюємо SIA один раз
    SIA = SentimentIntensityAnalyzer()
    SIA.lexicon.update(tone_dict)

    # Основна обробка
    news_df['sentiment_score'] = news_df['processed_text'].apply(
        lambda tokens: analyze_sentiment_from_tokens(tokens, SIA)
    )

    # Візуалізації
    fig_1 = plt.figure(figsize=(10, 6))
    plt.hist(news_df['sentiment_score'], bins=40, color='skyblue', edgecolor='black')
    plt.title("Розподіл тональності новин")
    plt.xlabel("Тональність (compound score)")
    plt.ylabel("Кількість новин")
    plt.show()

    sentiment_by_date = news_df.groupby(news_df['date'].dt.date)['sentiment_score'].mean()
    fig_2 = plt.figure(figsize=(10, 6))
    plt.plot(range(len(sentiment_by_date)), sentiment_by_date.values, marker='o', color='orange')
    plt.xticks(range(len(sentiment_by_date)), range(1, len(sentiment_by_date) + 1))
    plt.title("Середня тональність новин у часі")
    plt.xlabel("День")
    plt.ylabel("Середня тональність")
    plt.show()

    return fig_1, fig_2


# ------------------------- 4) Візуалізація згадок ключових осіб або подій за допомогою NER -------------------------
def extract_and_visualize_named_entities(news_df, nlp_ner_model):
    news_df['processed_text_str'] = news_df['processed_text'].apply(lambda tokens: " ".join(tokens))

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


# ------------------------- 5) Визначення копіпаст новин з використанням косинусової подібності -------------------------
def detect_repackaged_news(news_df, threshold=0.9, max_features=50000):
    # TF-IDF векторизація
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=None)
    tfidf_matrix = vectorizer.fit_transform(news_df['text'])

    # Косинусна подібність
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Відбір подібних пар
    similar_pairs = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[1]):
            sim_score = similarity_matrix[i, j]
            if sim_score >= threshold:
                title_i = news_df.loc[i, 'title']
                title_j = news_df.loc[j, 'title']
                if title_i != title_j:
                    similar_pairs.append({
                        'index_1': i,
                        'index_2': j,
                        'title_1': title_i,
                        'title_2': title_j,
                        'similarity': sim_score
                    })

    similar_df = pd.DataFrame(similar_pairs)

    # Побудова графіка як об'єкта Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(similar_df['similarity'], bins=20, kde=False, ax=ax)
    ax.set_title("Розподіл косинусної подібності серед подібних новин")
    ax.set_xlabel("Косинусна подібність")
    ax.set_ylabel("Кількість пар")
    ax.grid(True)
    fig.tight_layout()

    # Текстовий блок
    similarity_text_block = "## Виявлення схожих новин (репаковані тексти)\n"
    similarity_text_block += f"Загалом знайдено {len(similar_df)} пар новин з косинусною подібністю понад {threshold}.\n\n"
    similarity_text_block += "Найбільш схожі пари:\n\n"

    top_similar = similar_df.sort_values(by='similarity', ascending=False).head(10)
    for _, row in top_similar.iterrows():
        similarity_text_block += (
            f"- **{row['title_1']}**\n"
            f"  ↔ **{row['title_2']}** — подібність: {row['similarity']:.3f}\n"
        )

    return fig, similarity_text_block


# ------------------------- 6) Перевірка клікбейтності новин(порівняння заголовку і тексту) -------------------------
def analyze_title_text_similarity(news_df, max_features=50000):
    # Векторизація заголовків і текстів разом
    all_texts = news_df['title'].tolist() + news_df['text'].tolist()
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    n = len(news_df)
    title_vecs = tfidf_matrix[:n]
    text_vecs = tfidf_matrix[n:]

    # Обчислення діагоналі матриці подібності (заголовок ↔ текст)
    similarities = cosine_similarity(title_vecs, text_vecs).diagonal()
    news_df = news_df.copy()
    news_df['title_text_similarity'] = similarities

    # Побудова графіка
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(similarities, bins=50, kde=True, ax=ax)
    ax.set_title('Схожість між заголовками і текстами (Cosine Similarity)')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Кількість новин')
    ax.grid(True)
    fig.tight_layout()

    # Формування markdown-блоку
    markdown_text = "## Аналіз узгодженості заголовків і текстів\n"
    markdown_text += "Гістограма показує, наскільки заголовки відображають зміст текстів.\n"
    markdown_text += "Нижче наведено 10 новин із найменшою схожістю між заголовком і основним текстом — це потенційно клікбейт:\n\n"

    suspicious = news_df.sort_values(by='title_text_similarity').head(10)
    for _, row in suspicious.iterrows():
        markdown_text += f"- **{row['title']}** — схожість: {row['title_text_similarity']:.3f}\n"

    return fig, markdown_text


# ------------------------- 7) Перевірка маніпулятивності новини-------------------------
def analyze_manipulative_language(news_df, manipulative_words_path):
    # Завантаження словника
    with open(manipulative_words_path, "r", encoding="utf-8") as f:
        manipulative_lemmas = set(line.strip() for line in f if line.strip())

    # Функція підрахунку
    def count_manipulative_words(lemmatized_tokens, manip_words_set):
        return sum(1 for token in lemmatized_tokens if token in manip_words_set)

    # Копія датафрейму
    df = news_df.copy()

    # Обчислення абсолютної кількості
    df["manipulative_word_count"] = df["processed_text"].apply(
        lambda tokens: count_manipulative_words(tokens, manipulative_lemmas)
    )

    # Обчислення частки (нормалізація)
    df["manipulative_ratio"] = df.apply(
        lambda row: row["manipulative_word_count"] / len(row["processed_text"])
        if len(row["processed_text"]) > 0 else 0,
        axis=1
    )

    # Візуалізація
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["manipulative_ratio"], bins=30, kde=False, color="darkorange", ax=ax)
    ax.set_title("Розподіл частки маніпулятивних слів у новинах")
    ax.set_xlabel("Частка маніпулятивних слів")
    ax.set_ylabel("Кількість новин")
    ax.grid(True)
    fig.tight_layout()

    # Формування тексту
    markdown_text = "## Аналіз маніпулятивної лексики\n"
    markdown_text += (
        f"Загальна кількість слів у словнику: {len(manipulative_lemmas)}.\n\n"
        "Оцінено частку маніпулятивних лем у кожній новині. Нижче — заголовки з найбільшої кількістю співпадінь:\n\n"
    )

    top_manip = df.sort_values("manipulative_ratio", ascending=False).head(10)
    for _, row in top_manip.iterrows():
        markdown_text += (
            f"- **{row['title']}** — {row['manipulative_word_count']} маніпулятивних слів "
            f"({row['manipulative_ratio']:.2%})\n"
        )

    return fig, markdown_text
