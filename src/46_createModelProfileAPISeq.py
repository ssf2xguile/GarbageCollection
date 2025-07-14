"""
調査用データに対して正解APIメソッドシーケンスが何種類何回登場したのかをカウントするプログラム
入力: 元のテストデータのインデックスがついている調査用データと各モデルが予測したAPIメソッドシーケンスのテキストファイル
出力: jsonファイル (api_sequenceを配列として保存)
"""
import pandas as pd
import argparse
import json
import os
from collections import defaultdict

def parse_string_into_apis(str_):
    """
    予測ファイル一行の文字列をパースし、正規化されたAPIのリストを返す。
    例: ['api1', 'api2', ..., 'apin']
    """
    apis = []
    eles = str_.split('\t')[0].strip().split('.')

    if not eles or not eles[0]:
        return []

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ', 1)
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = library_
        except ValueError:
            module_ = eles[i].strip()
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = module_

    if first_lib and eles[-1]:
        api = first_lib.strip() + '.' + eles[-1].strip()
        api = api.lower().replace(' ', '')
        apis.append(api)
    
    return apis

def calculate_head_or_tail(sequence_stats_list):
    """
    APIシーケンスのリストを受け取り、出現回数に基づいてヘッドとテールを分類する
    """
    total_appearance_count = sum(entry["appearance_count"] for entry in sequence_stats_list)
    
    sorted_sequence_stats = sorted(sequence_stats_list, key=lambda x: x["appearance_count"], reverse=True)
    
    cumulative_count = 0
    head_threshold = total_appearance_count * 0.3
    
    for entry in sorted_sequence_stats:
        cumulative_count += entry["appearance_count"]
        if cumulative_count <= head_threshold:
            entry["head_or_tail"] = 0  # ヘッド
        else:
            entry["head_or_tail"] = 1  # テール

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the CodeBERT prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_predictions.txt', help="Path to the CodeT5 prediction.txt file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_predictions.txt', help="Path to the MulaRec prediction.txt file")
    parser.add_argument('--investigate_file', type=str, default='./data/target_investigate.csv', help="Path to the investigation target file")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the output files")
    args = parser.parse_args()

    # --- 1. データの読み込み ---
    # 各ファイルの予測結果をAPIのリストとして読み込む
    with open(args.file1, 'r', encoding='utf-8') as file1:
        codebert_preds = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds = [parse_string_into_apis(pred) for pred in file3.readlines()]

    df = pd.read_csv(args.investigate_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    # --- 2. 集計処理 ---
    # 【変更点】処理中は文字列キーを使い、元のリストを別の辞書で保持する
    sequence_key_to_list_map = {}
    sequence_stats = defaultdict(lambda: {
        "appearance_count": 0, "CodeBERT_appearance_count": 0, "CodeBERT_correct_count": 0,
        "CodeT5_appearance_count": 0, "CodeT5_correct_count": 0, "MulaRec_appearance_count": 0,
        "MulaRec_correct_count": 0,
    })

    for idx, correct_sequence_list in zip(original_index, correct_refs):
        pred_idx = int(idx) - 1
        codebert_sequence_list = codebert_preds[pred_idx]
        codet5_sequence_list = codet5_preds[pred_idx]
        mularec_sequence_list = mularec_preds[pred_idx]

        # リストを結合して、辞書のキーとして使える「文字列」を生成
        correct_key = ' . '.join(correct_sequence_list)
        codebert_key = ' . '.join(codebert_sequence_list)
        codet5_key = ' . '.join(codet5_sequence_list)
        mularec_key = ' . '.join(mularec_sequence_list)

        # 文字列キーと元のAPIリストの対応を保存
        sequence_key_to_list_map[correct_key] = correct_sequence_list
        sequence_key_to_list_map[codebert_key] = codebert_sequence_list
        sequence_key_to_list_map[codet5_key] = codet5_sequence_list
        sequence_key_to_list_map[mularec_key] = mularec_sequence_list
        
        # 文字列キーを使って統計情報を更新
        sequence_stats[correct_key]["appearance_count"] += 1
        sequence_stats[codebert_key]["CodeBERT_appearance_count"] += 1
        if codebert_key == correct_key:
            sequence_stats[codebert_key]["CodeBERT_correct_count"] += 1
        sequence_stats[codet5_key]["CodeT5_appearance_count"] += 1
        if codet5_key == correct_key:
            sequence_stats[codet5_key]["CodeT5_correct_count"] += 1
        sequence_stats[mularec_key]["MulaRec_appearance_count"] += 1
        if mularec_key == correct_key:
            sequence_stats[mularec_key]["MulaRec_correct_count"] += 1

    # --- 3. JSON出力用データ生成 ---
    sequence_stats_list = []
    # 【変更点】集計済みの文字列キー(api_sequence_key)でループ
    for api_sequence_key, stats in sequence_stats.items():
        # 文字列キーから元のAPIリストを取得し、"api_sequence"の値として設定
        entry = {"api_sequence": sequence_key_to_list_map.get(api_sequence_key, [])}
        entry.update(stats)
        sequence_stats_list.append(entry)

    sequence_stats_list = sorted(sequence_stats_list, key=lambda x: x["appearance_count"], reverse=True)
    calculate_head_or_tail(sequence_stats_list)

    # --- 4. 結果表示とファイル出力 ---
    head_count = sum(1 for entry in sequence_stats_list if entry["head_or_tail"] == 0)
    tail_count = sum(1 for entry in sequence_stats_list if entry["head_or_tail"] == 1)
    head_appearance_count = sum(entry["appearance_count"] for entry in sequence_stats_list if entry["head_or_tail"] == 0)
    tail_appearance_count = sum(entry["appearance_count"] for entry in sequence_stats_list if entry["head_or_tail"] == 1)

    print(f"ヘッドのAPIシーケンス: {head_count} 種類, 合計 {head_appearance_count} 件")
    print(f"テールのAPIシーケンス: {tail_count} 種類, 合計 {tail_appearance_count} 件")

    output_path = os.path.join(args.output_dir, 'sequence_correctCount_and_appearCount30.json')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(sequence_stats_list, outfile, indent=4, ensure_ascii=False)
    
    print(f"結果を {output_path} に出力しました。 (api_sequenceは配列形式)")

if __name__ == "__main__":
    main()