import requests
from bs4 import BeautifulSoup

# WebページのURLを指定
url = "https://e-singlemalt.co.jp/"

# URLからHTMLを取得
response = requests.get(url)

# レスポンスのHTMLをBeautifulSoupオブジェクトにパース
soup = BeautifulSoup(response.content, 'html.parser')

all_div = soup.find_all("div", class_="side-section-item-wrap")

for div in all_div:
    for a in div.find_all("a"):
        print(a.text)
    print("")