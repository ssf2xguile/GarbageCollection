import requests
from bs4 import BeautifulSoup

keywords = [
    "国土交通省", "介護施設", "国土交通省", "社会福祉法人", "東京都",
    "神奈川県", "埼玉県", "東京都", "特養", "特別養護老人ホーム",
    "横浜市", "令和6年", "北海道", "保育所", "広島", "埼玉県",
    "福祉施設", "神奈川", "千葉県", "北海道", "介護", "東京都",
    "国交省", "大阪府", "静岡", "介護老人保健施設", "介護施設"
]

# 1. ページのHTMLを取得
# 調べたいサイトのURL
url = f"https://soeikenso-hp.com/%E5%8E%9A%E6%9C%A8%E5%B8%82%E3%81%AE%E5%A4%A7%E8%A6%8F%E6%A8%A1%E4%BF%AE%E7%B9%95%E5%B7%A5%E4%BA%8B%E3%81%AB%E3%81%8A%E3%81%91%E3%82%8B%E7%89%B9%E5%BE%B4%E3%81%A8%E6%B3%A8%E6%84%8F%E7%82%B9/"  
response = requests.get(url)
html = response.content


# 2. BeautifulSoupでHTMLを解析
soup = BeautifulSoup(html, "html.parser")


# 3. <header>と<footer>を削除
if soup.header:
    soup.header.decompose()  # ヘッダーを削除
if soup.footer:
    soup.footer.decompose()  # フッターを削除


# 4. キーワードの出現回数をカウント
for keyword in keywords:
    text = soup.get_text()  # ページ全体のテキストを取得
    count = text.count(keyword)
    print(count)