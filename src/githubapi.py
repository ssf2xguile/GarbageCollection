import requests
import sys
from datetime import datetime

def parse_github_issue_url(url):
    # URLからリポジトリオーナー、リポジトリ名、issue番号を取得する
    parts = url.split('/')
    repo_owner = parts[3]
    repo_name = parts[4]
    issue_number = parts[6]
    return repo_owner, repo_name, issue_number

def get_issue_details(repo_owner, repo_name, issue_number):
    # GitHub APIを叩いてIssueの情報を取得する
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}"
    response = requests.get(api_url)
    issue_data = response.json()
    return issue_data

def get_issue_comments(repo_owner, repo_name, issue_number):
    # GitHub APIを叩いてIssueのコメントを取得する
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues/{issue_number}/comments"
    response = requests.get(api_url)
    comments = response.json()
    return comments

def format_issue_comments(comments):
    # Issueのコメントを指定された形式で整形する
    formatted_comments = []
    for comment in comments:
        user_name = comment['user']['login']
        created_at = datetime.strptime(comment['created_at'], "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
        content = comment['body']
        formatted_comments.append({
            'user_name': user_name,
            'created_at': created_at,
            'content': content
        })
    return formatted_comments

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python githubapi.py <GitHub Issue URL>")
        sys.exit(1)

    # コマンドライン引数からGitHub IssueのURLを取得
    github_issue_url = sys.argv[1]

    # URLを解析して必要な情報を取得
    repo_owner, repo_name, issue_number = parse_github_issue_url(github_issue_url)

    # GitHub APIを叩いてIssueの情報とコメントを取得
    issue_data = get_issue_details(repo_owner, repo_name, issue_number)
    issue_comments = get_issue_comments(repo_owner, repo_name, issue_number)

    # Issueタイトルとコメントを整形して出力
    issue_title = issue_data['title']
    issue_body = issue_data['body']
    formatted_comments = format_issue_comments(issue_comments)

    print(issue_title)
    print(f"{issue_data['user']['login']} commented at {issue_data['created_at']}")
    print(issue_body)
    print()
    for comment in formatted_comments:
        print(f"{comment['user_name']} commented at {comment['created_at']}")
        print(comment['content'])
        print()