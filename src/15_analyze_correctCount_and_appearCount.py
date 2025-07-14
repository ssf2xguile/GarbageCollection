"""
調査用データに対して正解APIメソッド(シーケンスの最初のAPIメソッド)が何種類何回登場したのかをカウントするプログラム
入力: 元のテストデータのインデックスがついている調査用データと各モデルが予測したAPIメソッドシーケンスのテキストファイル
出力: jsonファイル
"""
import pandas as pd
import argparse
import json
import os
from collections import defaultdict

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('\t')[0].strip().split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                api = first_lib.strip() + '.' + module_.strip()
                api = api.lower().replace(' ', '')
                apis.append(api)
                first_lib = library_
            except ValueError:
                module_ = eles[i].strip()
                library_ = ''
                api = first_lib.strip() + '.' + module_.strip()
                api = api.lower().replace(' ', '')
                apis.append(api)
                first_lib = module_

    api = first_lib.strip() + '.' + eles[-1].strip()
    api = api.lower().replace(' ', '')
    apis.append(api)
    return apis[0]

def calculate_head_or_tail(api_stats_list):
    total_appearance_count = sum(entry["first_appearance_count"] for entry in api_stats_list)
    
    # 出現回数で降順ソートし、上位50%をヘッド、それ以外をテールとする
    sorted_api_stats = sorted(api_stats_list, key=lambda x: x["first_appearance_count"], reverse=True)
    
    cumulative_count = 0
    head_threshold = total_appearance_count * 0.2  # 上位50%の閾値
    
    for entry in sorted_api_stats:
        cumulative_count += entry["first_appearance_count"]
        if cumulative_count <= head_threshold:
            entry["head_or_tail"] = 0  # ヘッド
        else:
            entry["head_or_tail"] = 1  # テール

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_predictions.txt', help="Path to save the output file")
    parser.add_argument('--investigate_file', type=str, default='./data/target_investigate.csv', help="Path to save the output file")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the text file which including tail data")
    args = parser.parse_args()

    # 予測結果を読み込む
    with open(args.file1, 'r', encoding='utf-8') as file1:
        codebert_preds = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds = [parse_string_into_apis(pred) for pred in file3.readlines()]

    # 調査対象データに該当する予測結果のインデックスのリストを取り出す
    df = pd.read_csv(args.investigate_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    # 結果を保持する辞書
    api_stats = defaultdict(lambda: {
        "first_appearance_count": 0,
        "CodeBERT_first_appearance_count": 0,
        "CodeBERT_correct_count": 0,
        "CodeT5_first_appearance_count": 0,
        "CodeT5_correct_count": 0,
        "MulaRec_first_appearance_count": 0,
        "MulaRec_correct_count": 0,
    })

    # 全モデルに共通するAPIメソッドを収集
    for idx, correct_api in zip(original_index, correct_refs):
        codebert_api = codebert_preds[int(idx)-1]
        codet5_api = codet5_preds[int(idx)-1]
        mularec_api = mularec_preds[int(idx)-1]

        # 正解APIメソッドの最初の登場回数をカウント
        api_stats[correct_api]["first_appearance_count"] += 1

        # CodeBERTの結果を反映
        api_stats[codebert_api]["CodeBERT_first_appearance_count"] += 1
        if codebert_api == correct_api:
            api_stats[codebert_api]["CodeBERT_correct_count"] += 1

        # CodeT5の結果を反映
        api_stats[codet5_api]["CodeT5_first_appearance_count"] += 1
        if codet5_api == correct_api:
            api_stats[codet5_api]["CodeT5_correct_count"] += 1

        # MulaRecの結果を反映
        api_stats[mularec_api]["MulaRec_first_appearance_count"] += 1
        if mularec_api == correct_api:
            api_stats[mularec_api]["MulaRec_correct_count"] += 1

    # 辞書をリスト形式に変換し、api_methodを一番上に配置
    api_stats_list = []
    for api_method, stats in api_stats.items():
        entry = {"api_method": api_method}
        entry.update(stats)
        api_stats_list.append(entry)

    # first_appearance_countで降順ソート
    api_stats_list = sorted(api_stats_list, key=lambda x: x["first_appearance_count"], reverse=True)

    # head_or_tail を計算して追加
    calculate_head_or_tail(api_stats_list)

    # ヘッドとテールのAPIメソッドの種類数と合計件数を集計
    head_count = sum(1 for entry in api_stats_list if entry["head_or_tail"] == 0)
    tail_count = sum(1 for entry in api_stats_list if entry["head_or_tail"] == 1)
    head_appearance_count = sum(entry["first_appearance_count"] for entry in api_stats_list if entry["head_or_tail"] == 0)
    tail_appearance_count = sum(entry["first_appearance_count"] for entry in api_stats_list if entry["head_or_tail"] == 1)

    # 結果を標準出力
    print(f"ヘッドのAPIメソッド: {head_count} 種類, 合計 {head_appearance_count} 件")
    print(f"テールのAPIメソッド: {tail_count} 種類, 合計 {tail_appearance_count} 件")

    # JSONファイルに出力
    with open(os.path.join(args.output_dir, 'first_correctCount_and_appearCount20.json'), 'w', encoding='utf-8') as outfile:
        json.dump(api_stats_list, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()