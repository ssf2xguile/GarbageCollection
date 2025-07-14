import requests
from bs4 import BeautifulSoup

# WebページのURLを指定
url = "https://ultimatespirits.jp/shopbrand/americanwhiskey/"

# URLからHTMLを取得
response = requests.get(url)

# レスポンスのHTMLをBeautifulSoupオブジェクトにパース
soup = BeautifulSoup(response.content, 'html.parser')
div = soup.find("div", class_="section sub-category")

for a in div.find_all("a"):
    print(a.text)
