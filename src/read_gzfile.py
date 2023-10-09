import gzip
import json

# gzファイルのパスを指定
file_path = 'context-ratio_0.5_without-history.jsonl.gz'

try:
    # gzファイルを開く
    print(f'ファイル名: {file_path} を読み込みます。')
    with gzip.open(file_path, 'rt') as file:
        # 最初の100行を読み取る
        for i, line in enumerate(file, 1):
            print(line)
            if i >= 30:
                break
except FileNotFoundError:
    print(f'ファイル {file_path} が見つかりません。')
except Exception as e:
    print(f'エラーが発生しました: {e}')            