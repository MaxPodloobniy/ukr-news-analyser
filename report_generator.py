import base64
from io import BytesIO
import datetime
from jinja2 import Template


def fig_to_base64(fig):
    """Конвертує matplotlib figure в base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    return img_str


def generate_analytics_report(df, figures, texts):
    """
    Генерує HTML звіт з результатами аналізу новин

    Parameters:
    -----------
    df : pandas DataFrame
        Датафрейм з новинами
    figures : dict
        Словник з matplotlib figures
    texts: dict
        Словник з текстовими результатами аналізу
    """
    # Конвертуємо всі графіки в base64
    encoded_figures = {
        name: fig_to_base64(fig)
        for name, fig in figures.items()
    }

    # HTML шаблон для звіту
    template_string = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Аналітичний звіт по новинам</title>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .section {
                margin-bottom: 40px;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                padding: 20px 0;
                margin-bottom: 30px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
                margin-top: 0;
            }
            .plot {
                text-align: center;
                margin: 20px 0;
            }
            .plot img {
                max-width: 100%;
                height: auto;
            }
            .metadata {
                color: #666;
                font-size: 0.9em;
                margin-bottom: 20px;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .summary {
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .summary ul {
                list-style-type: none;
                padding-left: 0;
            }
            .summary li {
                margin-bottom: 10px;
                padding-left: 20px;
                position: relative;
            }
            .summary li:before {
                content: "•";
                position: absolute;
                left: 0;
                color: #3498db;
            }
        </style>
    </head>
    <body>
        <h1>Аналітичний звіт по новинам</h1>

        <div class="metadata">
            <p><strong>Дата створення:</strong> {{ date }}</p>
            <p><strong>Проаналізовано новин:</strong> {{ total_news }}</p>
            <p><strong>Період аналізу:</strong> {{ date_range }}</p>
        </div>

        <div class="summary">
            <h2>Короткий огляд</h2>
            <ul>
                <li><strong>Загальна кількість публікацій:</strong> {{ total_news }}</li>
                <li><strong>Середня тональність:</strong> {{ avg_sentiment }}</li>
            </ul>
        </div>

        <div class="section">
            <h2>1. Частота публікацій</h2>
            <p>Аналіз розподілу публікацій за часом:</p>
            <div class="plot">
                <img src="data:image/png;base64,{{ figures.publication_freq }}" alt="Publication Frequency">
            </div>
        </div>

        <div class="section">
            <h2>2. Хмара слів</h2>
            <p>Візуалізація найчастіше вживаних слів:</p>
            <div class="plot">
                <img src="data:image/png;base64,{{ figures.wordcloud }}" alt="Word Cloud">
            </div>
        </div>

        <div class="section">
            <h2>3. Тематичний аналіз (LDA)</h2>
            <p>Основні теми, виявлені в новинах:</p>
            <ul>
                {% for idx, topic in texts.formed_topics %}
                <li>Тема {{ idx+1 }}: {{ topic }}</li>
                {% endfor %}
            </ul>
        </div>

        <div class="section">
            <h2>4. Аналіз тональності</h2>
            <p>Розподіл емоційного забарвлення новин:</p>
            <div class="plot">
                <img src="data:image/png;base64,{{ figures.all_tonality }}" alt="Sentiment Analysis">
            </div>
            <p>Розподіл тональності за часом:</p>
            <div class="plot">
                <img src="data:image/png;base64,{{ figures.tonality_per_time }}" alt="Sentiment Analysis">
            </div>
        </div>

        <div class="section">
            <h2>5. Аналіз згадувань (NER)</h2>
            <p>Частота згадування ключових осіб та організацій:</p>
            <div class="plot">
                <img src="data:image/png;base64,{{ figures.ner_visualization }}" alt="NER Analysis">
            </div>
        </div>
    </body>
    </html>
    """

    # Підготовка даних для шаблону
    template_data = {
        'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_news': len(df),
        'date_range': f"{df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}",
        'avg_sentiment': f"{df['sentiment_score'].mean():.2f}",
        'figures': encoded_figures,
        'texts': {
            'formed_topics': texts['formed_topics']
        }
    }

    # Створення звіту
    template = Template(template_string)
    html_report = template.render(**template_data)

    # Збереження звіту
    with open('news_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)

    return 'news_analysis_report.html'
