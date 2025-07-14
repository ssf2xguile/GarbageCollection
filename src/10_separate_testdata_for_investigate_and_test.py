"""
テストデータ(test_3_lines.csv)を調査用とテスト用にわける。
1:1の割合になるように分割する。
"""
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def main():
    # 引数の設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='./data/test_3_lines.csv', help="Path to the test file")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the output files")
    args = parser.parse_args()

    # CSVファイルを読み込み
    df = pd.read_csv(args.test_file)
    
    # インデックスカラムを追加（1から始まる）  元のテストデータの何行目かを確認できるようにするため
    df['api_index'] = range(1, len(df) + 1)
    
    # ランダムにデータを1:1で分割（train_test_splitを使ってランダム分割）
    target_investigate, target_test = train_test_split(df, test_size=0.5, random_state=42)

    # 調査用とテスト用データをCSVファイルに出力
    target_investigate.to_csv(os.path.join(args.output_dir, 'target_investigate.csv'), index=False)
    target_test.to_csv(os.path.join(args.output_dir, 'target_test.csv'), index=False)

    print("Data has been split and saved successfully.")

if __name__ == "__main__":
    main()