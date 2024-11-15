"""
Тут парсимо дані з сайтів.
Перший парсер з сайту Укр Правди, в неї є основна сторінка с усіма публікаціями за останній час,
на цій сторінці ми парсимо посилання на публікації і час публікації, поки час не перевищує дві доби ми
продовжуємо парсити переходячи за посилання і збираючи весь текст публікацій.
Другий парсер з сайту Бабеля, в них є сторінка text-sitemap де можна отримати посилання на всі новини в
конктретний день і структура посилання доволі зрозуміла наприклад 'https://babel.ua/text-sitemap/2024-11/12'
отже ми парсимо спочатку цю сторінку потім по черзі обходимо всі посилання і парсимо звідти текст новин
Третій парсер з сайту RBC, в них теж в архіві посилання складається з дати і повертає всі новини за визначений
день наприклад 'https://www.rbc.ua/rus/archive/2024/11/10' ну і знову парсимо цю сторінку потім переходимо за
посиланнями і парсимо за посиланнями всі тексти
Четвертий парсер з сайту кореспондету, в них структура сайту доволі схожа на РБК але в них новини кожного дня розділені
ще й на декілька сторінок і треба пройти їх всі, ось стандартне посиланння на РБК:
'https://ua.korrespondent.net/all/2024/november/11/p8/'
"""
import requests
import re
from requests_html import HTMLSession
from bs4 import BeautifulSoup as bs
import pandas as pd
from datetime import datetime, timedelta

# Словник з українськими місяцями, треба бо datetime не розуміє укр. місяці
ukrainian_months = {
    "січня": "January", "лютого": "February", "березня": "March", "квітня": "April",
    "травня": "May", "червня": "June", "липня": "July", "серпня": "August",
    "вересня": "September", "жовтня": "October", "листопада": "November", "грудня": "December"
}

# Заголовки, що імітують реальний браузер
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://censor.net/",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1"
}

headers_2 = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

# Скільки днів парсимо
days_to_parse = 7


def get_html_file(url, render_js=False):
    """Повертає об'єкт BeautifulSoup з HTML сторінки. Виконує рендеринг JavaScript, якщо render_js=True."""
    if render_js:
        session = HTMLSession()
        response = session.get(url, headers=headers)
        response.html.render(timeout=20)  # Рендеринг JS, збільшуємо таймаут для надійності
        html_content = response.html.html
    else:
        response = requests.get(url, headers=headers_2)
        response.encoding = 'utf-8'
        if response.status_code != 200:
            raise RuntimeError(f"HTTP помилка: {response.status_code}, сайт {url}")
        html_content = response.text

    return bs(html_content, 'html.parser')


def parse_ukr_pravda():
    print('\n Парсинг сайту pravda.com.ua')
    url = 'https://www.pravda.com.ua/articles/'
    soup = get_html_file(url)

    news_data_list = []  # Зберігатимемо всі дані в списку словників

    now = datetime.now()
    two_days_ago = now - timedelta(days=days_to_parse)

    articles = soup.find_all("div", attrs={'class': 'article article_list'})

    for article in articles:
        # Спочатку знаходимо дату публікації
        date_text = article.find("div", attrs={'class': 'article_author'}).text.split("—")[0].strip()

        # Замінити українські місяці на англійські
        for ukr_month, eng_month in ukrainian_months.items():
            date_text = date_text.replace(ukr_month, eng_month)

        # Конвертувати рядок дати у datetime об'єкт
        date = datetime.strptime(date_text, "%d %B %Y, %H:%M")

        # Якщо дата публікації свіжіша за 2 дні
        if date >= two_days_ago:
            header = article.find("div", attrs={'class': 'article_header'}).find("a")
            title = header.text.strip()
            article_url = header['href']

            article_soup = get_html_file(article_url)
            text_body = article_soup.find('div', attrs={"class": ["post__text", "post_text", "post_article_text"]})
            if text_body:
                # Видаляємо небажані елементи
                for unwanted in text_body.find_all(['script', 'style', 'blockquote']):
                    unwanted.decompose()

                # Збираємо тільки унікальні параграфи
                paragraphs = []
                seen_texts = set()

                for p in text_body.find_all("p"):
                    text = p.text.strip()
                    if text and text not in seen_texts:  # Перевіряємо на унікальність
                        seen_texts.add(text)
                        paragraphs.append(text)

                content = " ".join(paragraphs)

                print(content)
                # Додаткова перевірка на пустий контент
                if content.strip():
                    news_data_list.append({
                        "title": title,
                        "date": date,
                        "text": content,
                        "url": article_url
                    })
        else:
            # Так як на сайті новини відсортовані по часу можна до кінця сторінки і не доходити
            break

    print(f'Знайдено {len(news_data_list)} статей')
    news_data = pd.DataFrame(news_data_list)
    return news_data


def parse_babel():
    print('\n Парсинг сайту babel.ua')
    url = 'https://babel.ua/text-sitemap/'

    curr_date = datetime.today()
    curr_year = curr_date.year
    curr_month = curr_date.month
    curr_day = curr_date.day

    news_data_list = []  # Зберігатимемо всі дані в списку словників

    for i in range(days_to_parse):
        print(f'Парсимо новини за день {curr_day:02d}, місяць {curr_month:02d}, рік {curr_year}')

        curr_url = f"{url}{curr_year}-{curr_month:02d}/{curr_day:02d}"
        soup = get_html_file(curr_url)

        # Знаходимо останній div на сторінці, де містяться посилання на новини
        last_div = soup.find_all("div")[-1]

        # Перебираємо всі посилання в цьому div і збираємо інформацію
        for li in last_div.find_all("li"):
            a_tag = li.find("a")

            if a_tag:
                curr_soup = get_html_file(a_tag['href'])

                # Отримуємо дату
                time_element = curr_soup.find('time')
                date_str = time_element.get('datetime')
                date_time = datetime.fromisoformat(date_str).replace(tzinfo=None) # Перетворюємо на datetime і видаляємо час. пояс

                # З'єднуємо весь текст з параграфів
                content_div = curr_soup.find("div", attrs={'class': 'c-post-text'})  # Блок з текстом новини
                paragraphs = content_div.find_all("p") if content_div else []
                news_text = "\n".join(p.text.strip() for p in paragraphs)

                news_data_list.append({
                    "title": a_tag.text.strip(),
                    "date": date_time,
                    "text": news_text,
                    "url": a_tag['href']
                })
        curr_day -= 1

    news_data = pd.DataFrame(news_data_list)
    print(f'Всього знайдено {news_data.shape[0]} статей')

    return news_data


def parse_rbc():
    print('\n Парсинг сайту rbc.ua')

    base_url = 'https://www.rbc.ua/rus/archive/'

    news_data_list = []  # Зберігатимемо всі дані в списку словників

    curr_date = datetime.today()
    curr_year = curr_date.year
    curr_month = curr_date.month
    curr_day = curr_date.day

    for i in range(days_to_parse):
        print(f'Парсимо новини за день {curr_day:02d}, місяць {curr_month:02d}, рік {curr_year}')
        curr_url = f"{base_url}{curr_year}/{curr_month:02d}/{curr_day:02d}"
        soup = get_html_file(curr_url)

        body_div = soup.find('div', attrs={'class': 'newsline'})
        news = body_div.find_all('div')

        for article in news:
            href = article.find('a')['href']
            title = article.find('a').text.strip().splitlines()[-1] # Треба бо там одразу час і після \n заголовок

            # Додаємо рік, місяць і день до часу, конвертуємо в формат datetime
            article_time = article.find('span').text.strip()  # Наприклад, "04:03"
            full_datetime_str = f"{curr_year}-{curr_month:02d}-{curr_day:02d} {article_time}"
            full_datetime = datetime.strptime(full_datetime_str, "%Y-%m-%d %H:%M")

            # Отримуємо текст з новини
            soup = get_html_file(href)
            full_body_with_text = soup.find('div', attrs={'class': 'txt'})
            full_text = ' '.join([tag.get_text() for tag in full_body_with_text.find_all(['p', 'h2', 'li'])])

            news_data_list.append({
                "title": title,
                "date": full_datetime,
                "text": full_text,
                "url": href
            })

        curr_day -= 1
    news_data = pd.DataFrame(news_data_list)
    print(f'Всього знайдено {news_data.shape[0]} статей')

    return news_data


def parse_korrespondent():
    print('\n Парсинг сайту korrespondent.net')
    base_url = 'https://ua.korrespondent.net/all/'
    news_data_list = []

    curr_date = datetime.today()
    curr_year = curr_date.year
    curr_month_literal = curr_date.strftime("%B").lower()
    curr_month = curr_date.month
    curr_day = curr_date.day

    # Для кожного дня
    for q in range(days_to_parse):
        print(f'Парсимо новини за день {curr_day:02d}, місяць {curr_month:02d}, рік {curr_year}')
        curr_url = f'{base_url}{curr_year}/{curr_month_literal}/{curr_day:02d}/'

        # Проходимся по кожній сторінці за цей день поки не почнуться пусті сторінки(закінчаться новини)
        for i in range(1, 50):
            curr_page_url = f'{curr_url}p{i}/'

            # Отримуєм весь HTML сторінки
            soup = get_html_file(curr_page_url, render_js=False)
            articles_list = soup.find('div', attrs={'class': 'articles-list'})

            # Якщо сторінка не пуста
            if articles_list.contents != ['\n']:
                # Формуємо списки посилань на новини і з заголовками з сторінки
                titles = []
                href_list = []

                for article_title in articles_list.find_all('div', attrs={'class': 'article__title'}):
                    link = article_title.find('a')
                    titles.append(link.text.strip())
                    href_list.append(link.get('href'))

                # Формуємо дати публікацій всіх новин на сторінці в форматі datetime
                dates = [date.text for date in articles_list.find_all('div', attrs={'class': 'article__date'})]

                time_pattern = r'(\d{2}):(\d{2})'
                datetime_list = []

                for date_str in dates:
                    match = re.search(time_pattern, date_str)
                    full_datetime_str = f"{curr_year}-{curr_month:02d}-{curr_day:02d} {match.group(0)}"
                    full_datetime = datetime.strptime(full_datetime_str, "%Y-%m-%d %H:%M")
                    datetime_list.append(full_datetime)

                # Проходимось за посиланнями по всім новинам і збираємо з них весь текст
                article_texts_list = []

                for href in href_list:
                    article_soup = get_html_file(href)

                    article_raw_text = article_soup.find('div', attrs={'class': 'post-item__text'})
                    full_text = ' '.join([tag.get_text() for tag in article_raw_text.find_all(['p', 'h2', 'li'])])
                    full_text = ' '.join(full_text.split('\n')[:-2]) # Видаляємо останні два абзаци бо там реклама

                    article_texts_list.append(full_text)

                news_data_list.append({
                    "title": titles,
                    "date": datetime_list,
                    "text": article_texts_list,
                    "url": href_list
                })
            else:
                break

        curr_day -= 1

    # Створюємо DataFrame із списку словників
    news_df = pd.DataFrame(news_data_list)

    # Розгортаємо списки в окремі рядки
    news_df = news_df.explode(['title', 'url', 'text', 'date'], ignore_index=True)

    print(f'Всього знайдено {news_df.shape[0]} статей')

    return news_df


def parse_all_sites():
    # ukr_pravda_df = parse_ukr_pravda()
    babel_df = parse_babel()
    rbc_df = parse_rbc()
    korrespondent_df = parse_korrespondent()

    print(babel_df['date'])

    all_articles_df = pd.concat([babel_df, rbc_df, korrespondent_df], axis=0, ignore_index=True)
    all_articles_df['text'] = all_articles_df['text'].str.replace(r'[\n\t\r]', ' ', regex=True)

    all_articles_df.to_csv('parsed_articles.csv',
                           index=False,
                           encoding='utf-8',
                           quoting=1)

    print(f'\nВсього за період в {days_to_parse} було знайдено {all_articles_df.shape[0]}')


parse_all_sites()
