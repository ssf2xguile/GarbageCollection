"""
調査用データに対して正解APIメソッド(シーケンスの最初のAPIメソッド)が何種類何回登場したのかをカウントするプログラム
入力: 元のテストデータのインデックスがついている調査用データ
出力: jsonファイル
"""
import pandas as pd
import argparse
import json
import os
from collections import Counter

def parse_string_into_apis(str_):   # 今回は小文字化と空白削除の処理を追加して、さらに最初の1個のAPIメソッドだけしか返さない。つまり、文字列が返ってくる。
    apis = []
    eles = str_.split('\t')[0].strip().split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            api = first_lib.strip()+'.'+module_.strip()
            api = api.lower().replace(' ','')
            apis.append(api)
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                api = first_lib.strip()+'.'+module_.strip()
                api = api.lower().replace(' ','')
                apis.append(api)
                first_lib = library_
            except ValueError:  # splitで失敗した場合の処理 例えばPoint . Math . pow Math . sqrtのようにドットが繋がっている場合
                module_ = eles[i].strip()
                library_ = ''
                api = first_lib.strip()+'.'+module_.strip()
                api = api.lower().replace(' ','')
                apis.append(api)
                first_lib = module_

    api = first_lib.strip()+'.'+eles[-1].strip()
    api = api.lower().replace(' ','')
    apis.append(api)
    return apis[0]  


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--investigate_file', type=str, default='./data/target_investigate.csv', help="Path to save the output file")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the text file which including tail data")
    args = parser.parse_args()

    # 調査対象データに該当する予測結果のインデックスのリストを取り出す
    df = pd.read_csv(args.investigate_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()  # この処理によってリストのリストが得られる[...,'base64.decodetoobject', 'math.sqrt', 'thread.start',...]
    
    # APIの出現頻度をカウント
    vocab = Counter(correct_refs)
    vocab_comm = vocab.most_common()
    json_output = []
    for api, count in vocab_comm:
        json_output.append({"api_method": api, "first_appearance_count": count})
    
    with open(os.path.join(args.output_dir, 'first_correct_api.json'), 'w') as json_file:
        json.dump(json_output, json_file, indent=4)
    
    print("JSON file 'first_correct_api.json' has been created.")

if __name__ == "__main__":
    main()