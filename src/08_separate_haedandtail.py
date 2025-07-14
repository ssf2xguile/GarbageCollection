import json
import pandas as pd
import argparse
import os

def calculate_label_freq(task, data_dir, output_dir):
    ### API class Data
    splits = ['train', 'test', 'validate']
    splits = ['test']
    all_data = []
    
    # データ読み込み
    for split in splits:
        file_name = data_dir + split + '_3_lines.csv'
        df = pd.read_csv(file_name)
        all_data.append(df)
    pd_all_data = pd.concat(all_data)
    all_data = list(pd_all_data['target_api'])

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
        all_api_data.append(api_seqs)                               # リストのリストとして追加
    
    # head_api_list.jsonとtail_api_list.jsonの読み込み
    with open(os.path.join(data_dir, 'api_head_or_tail.json'), 'r') as head_file:
        api_list = json.load(head_file)

    # APIメソッドとその対応する情報（head_or_tail）を辞書に変換
    api_dict = {entry['api_method']: entry['head_or_tail'] for entry in api_list}

    # 結果を保存する配列
    head_api_index = []
    tail_api_index = []

    # APIを1つずつ確認して、対応する配列に追加  APIシーケンスの最初のAPIメソッドがテールであれば、テールAPIシーケンスとして扱う
    for idx, api_seq in enumerate(all_api_data):
        is_head = True
        if api_seq[0] in api_dict and api_dict[api_seq[0]] == 1:  # 辞書を使用して head_or_tail の情報を確認
            is_head = False
        if is_head:
            head_api_index.append(idx+1)    # 何行目のデータかを示すため、インデックスは1から始める必要がある
        else:
            tail_api_index.append(idx+1)

    # head_api_indexとtail_api_indexに基づいて、pd_all_dataから対応するデータを抽出
    pd_head_data = pd_all_data.iloc[head_api_index]  # head APIに対応するデータを抽出
    pd_tail_data = pd_all_data.iloc[tail_api_index]  # tail APIに対応するデータを抽出

    # head_api_indexとtail_api_indexをそれぞれのデータに新しい列として追加
    pd_head_data['api_index'] = head_api_index
    pd_tail_data['api_index'] = tail_api_index

    # 確認のために結果を出力
    print(f"Test HEAD API Seq: {len(pd_head_data)}")
    print(f"Test TAIL API Seq: {len(pd_tail_data)}")

    # CSVファイルとして保存（インデックス情報を含む）
    pd_head_data.to_csv(os.path.join(output_dir, 'test_head_lines.csv'), index=False)
    pd_tail_data.to_csv(os.path.join(output_dir, 'test_tail_lines.csv'), index=False)

def main():
    task = 'api'
    data_dir = './data/'
    output_dir = './data/'
    calculate_label_freq(task, data_dir, output_dir)

if __name__ == "__main__":
    main()

