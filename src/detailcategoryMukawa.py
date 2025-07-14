import requests
from bs4 import BeautifulSoup

# WebページのURLを指定
url = "https://mukawa-spirit.com/?mode=f32"

# URLからHTMLを取得
response = requests.get(url)

# レスポンスのHTMLをBeautifulSoupオブジェクトにパース
soup = BeautifulSoup(response.content, 'html.parser')
#print(soup.prettify())
ul = soup.find('ul', class_='products-index-list-02')
a_list = ul.find_all('a')
# href属性の値を表示

for a_href in a_list:
    print(a_href.text.strip())