import os
import datetime


def save_figure(fig, filename):
    """Зберігає графік у файл."""
    fig_path = os.path.join("reports", filename)
    fig.savefig(fig_path, format='png', bbox_inches='tight')
    return fig_path

def generate_markdown_report(df, figures, texts):
    """
    Генерує Markdown-звіт із результатами аналізу новин.

    Parameters:
    -----------
    df : pandas DataFrame
        Датафрейм із новинами.
    figures : dict
        Словник із matplotlib figures.
    texts: dict
        Словник із текстовими результатами аналізу.
    """
    os.makedirs("reports", exist_ok=True)

    # Збереження графіків
    figure_paths = {
        name: save_figure(fig, f"{name}.png")
        for name, fig in figures.items()
    }

    # Створення тексту звіту
    markdown_content = f"# Аналітичний звіт по новинам\n\n"
    markdown_content += f"**Дата створення:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    markdown_content += f"**Проаналізовано новин:** {len(df)}\n\n"
    markdown_content += f"**Період аналізу:** {df['date'].min().strftime('%Y-%m-%d')} - {df['date'].max().strftime('%Y-%m-%d')}\n\n"
    markdown_content += f"**Середня тональність:** {df['sentiment_score'].mean():.2f}\n\n"

    # Додати секції
    markdown_content += "## Частота публікацій\n"
    markdown_content += "Аналіз розподілу публікацій за часом:\n\n"
    markdown_content += f"![Частота публікацій]({figure_paths['publication_freq']})\n\n"

    markdown_content += "## Хмара слів\n"
    markdown_content += "Найчастіше вживані слова:\n\n"
    markdown_content += f"![Хмара слів]({figure_paths['wordcloud']})\n\n"

    markdown_content += "## Тематичний аналіз (LDA)\n"
    markdown_content += "Основні теми:\n\n"
    for idx, topic in texts['formed_topics']:
        markdown_content += f"- **Тема {idx + 1}:** {topic}\n"

    markdown_content += "\n## Аналіз тональності\n"
    markdown_content += "Розподіл емоційного забарвлення:\n\n"
    markdown_content += f"![Аналіз тональності]({figure_paths['all_tonality']})\n\n"
    markdown_content += "Розподіл тональності за часом:\n\n"
    markdown_content += f"![Тональність за часом]({figure_paths['tonality_per_time']})\n\n"

    markdown_content += "## Аналіз згадувань (NER)\n"
    markdown_content += "Частота згадування ключових осіб та організацій:\n\n"
    markdown_content += f"![NER Аналіз]({figure_paths['ner_visualization']})\n\n"

    # Збереження звіту
    with open("news_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
