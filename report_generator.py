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

    # Секція: Частота публікацій
    markdown_content += "## Частота публікацій\n"
    markdown_content += (
        "Цей графік відображає, як змінювалась кількість новин з часом. "
        "Піки можуть свідчити про події, що привернули велику увагу ЗМІ, тоді як провали — про інформаційне затишшя.\n\n"
    )
    markdown_content += f"![Частота публікацій]({figure_paths['publication_freq']})\n\n"

    # Секція: Хмара слів
    markdown_content += "## Хмара слів\n"
    markdown_content += (
        "Хмара слів показує, які лексеми найчастіше зустрічались у текстах новин. "
        "Великі слова — часті, дрібні — рідкісні. Це дозволяє побачити основну тематику публікацій.\n\n"
    )
    markdown_content += f"![Хмара слів]({figure_paths['wordcloud']})\n\n"

    # Секція: Аналіз тональності
    markdown_content += "## Аналіз тональності\n"
    markdown_content += (
        "Тональність — це оцінка емоційного забарвлення тексту (негативна, нейтральна, позитивна). "
        "Розподіл показує загальну емоційну атмосферу в ЗМІ. Також аналіз подає, як змінювалась тональність із часом.\n\n"
    )
    markdown_content += f"![Аналіз тональності]({figure_paths['all_tonality']})\n\n"
    markdown_content += f"![Тональність за часом]({figure_paths['tonality_per_time']})\n\n"

    # Секція: NER
    markdown_content += "## Аналіз згадувань (NER)\n"
    markdown_content += (
        "Цей розділ показує, які іменовані сутності (особи, організації, географічні назви) згадуються найчастіше. "
        "Це дає змогу оцінити фокус ЗМІ на ключових фігурах і темах.\n\n"
    )
    markdown_content += f"![NER Аналіз]({figure_paths['ner_visualization']})\n\n"

    # Секція: Repackaged news
    markdown_content += "## Виявлення схожих новин\n"
    markdown_content += (
        "За допомогою векторизації та кластеризації виявлялись новини з дуже схожим текстом. "
        "Це дозволяє знайти копії новин, розміщені на різних сайтах або варіації одних і тих же повідомлень.\n\n"
    )
    markdown_content += f"![Схожість новин]({figure_paths['repackaged_news']})\n\n"
    markdown_content += texts['repackaged_news']

    # Секція: Заголовки vs текст
    markdown_content += "## Узгодженість заголовків і текстів\n"
    markdown_content += (
        "Цей аналіз вимірює схожість між заголовками та основним текстом новини. "
        "Низька схожість може свідчити про клікбейт або маніпуляцію. "
        "Показано як загальну картину, так і приклади сумнівних новин.\n\n"
    )
    markdown_content += f"![Схожість заголовок ↔ текст]({figure_paths['title_text_similarity']})\n\n"
    markdown_content += texts['title_text_similarity']

    # Секція: Маніпуляція
    markdown_content += "## Маніпулятивна лексика в новинах\n"
    markdown_content += (
        "Оцінено частоту використання слів, які можуть вказувати на маніпулятивну риторику "
        "(емоційно забарвлені оцінки, тиск, узагальнення тощо). "
        "Це дає змогу виявити публікації з потенційно навмисним впливом на читача.\n\n"
    )
    markdown_content += f"![Маніпулятивність]({figure_paths['manipulative_language']})\n\n"
    markdown_content += texts['manipulative_language']

    # Збереження
    with open("news_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Збереження звіту
    with open("news_analysis_report.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
