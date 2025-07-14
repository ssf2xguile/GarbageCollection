"""
学習データ(train_3_lines.csv)から全てのAPIメソッドを抽出して、ラベル付けを行い、APIメソッドがヘッドかテールかを判別するプログラム
RQ4に合わせてヘッドを0, テールを1とみなす
入力: train3_lines.csv, 出力: api_head_or_tail.json
"""
import json
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import argparse
import os

def calculate_label_freq(task, data_dir, output_dir):
    ###API class Data
    splits = ['train', 'test','validate']
    splits = ['test']
    all_data = []
    for split in splits:
        file_name = data_dir + split + '_3_lines.csv'
        df = pd.read_csv(file_name)
        all_data.append(df)
        print(len(df))
    all_data = pd.concat(all_data)
    all_data = list(all_data['target_api'])
    print(len(all_data))

    all_api_data = []
    def parse_string_into_apis(str_):
        apis = []
        eles = str_.split('.')

        first_lib = eles[0]

        for i in range(1, len(eles)-1):
            try:
                module_, library_ = eles[i].strip().rsplit(' ')
            except:
                module_, library_ = eles[i].strip().split(' ', 1)
            apis.append(first_lib.strip()+'.'+module_.strip())
            first_lib = library_

        apis.append(first_lib.strip() +'.'+ eles[-1].strip())
        return apis

    for d in all_data:
        api_seqs = parse_string_into_apis(d)                        # APIメソッドシーケンスの抽出
        api_seqs = [a.lower().replace(' ', '') for a in api_seqs]   # 小文字化と空白の削除を行う
        all_api_data.extend(api_seqs)                               # appendではないのは追加するのがリストの中身であるAPIメソッドであるから。appendにしてしまうとリスト化されているAPIメソッドシーケンスが追加されてしまう
    all_data = all_api_data
    # print(all_data[:20]) 例['string.length', 'string.charat', 'string.replace', 'string.length', 'string.charat', 'string.substring', 'string.format', 'string.substring', 'jscrollpane.getheight', 'jscrollpane.getverticalscrollbar', 'jscrollpane.getwidth', 'jscrollpane.gethorizontalscrollbar', 'scanner.uselocale', 'vector < string >.elementat', 'defaulttablemodel.<init>', 'jtable.setmodel', 'jscrollpane.setviewportview', 'jbutton.settext', 'mouseadapter.<init>', 'jbutton.addmouselistener']

    # APIの出現頻度をカウント
    vocab = Counter(all_api_data)
    vocab_comm = vocab.most_common()
    total_count = sum(vocab.values())
    
    # APIメソッドの種類数を表示
    print(f"APIメソッドの種類数: {len(vocab)}")

    # 上位50%までのAPIメソッドをヘッドとし、残りをテールに分類
    cumulative_count = 0
    threshold_count = total_count / 2
    json_output = []

    for api, count in vocab_comm:
        cumulative_count += count
        head_or_tail = 0 if cumulative_count <= threshold_count else 1
        json_output.append({"api_method": api, "appearance_count": count, "head_or_tail": head_or_tail})
    
    with open(os.path.join(output_dir, 'api_head_or_tail.json'), 'w') as json_file:
        json.dump(json_output, json_file, indent=4)
    
    print("JSON file 'api_head_or_tail.json' has been created.")
    
def main():
    task='api'
    data_dir='./data/'
    output_dir='./data/'
    calculate_label_freq(task, data_dir, output_dir)

if __name__ == "__main__":
    main()