"""
訓練データと各モデルの予測結果を比較し、正解APIメソッド（シーケンスの最初のAPI）の登場回数や
各モデルの正解数を集計するプログラム。

入力:
- 元の訓練データ (CSV形式)
- 各モデルが予測したAPIメソッドシーケンスのテキストファイル (3つ)

処理:
- 各ファイルの先頭10万件を対象とする。
- 正解APIの登場回数をカウントし、ヘッド/テールに分類する。
- 各モデルの予測APIの登場回数と正解数をカウントする。

出力:
- 各APIメソッドに関する統計情報をまとめたJSONファイル。
"""
import pandas as pd
import argparse
import json
import os
from collections import defaultdict

def parse_string_into_apis(str_):
    """
    タブ区切りの予測文字列から最初のAPIメソッドを抽出する。
    """
    if not isinstance(str_, str):
        return ""
        
    apis = []
    # 最初のタブ以前の部分（最初の予測候補）のみを使用
    prediction_part = str_.split('\t')[0]
    eles = prediction_part.strip().split('.')

    if len(eles) < 2:
        return prediction_part.strip().lower().replace(' ', '')

    first_lib = eles[0]

    for i in range(1, len(eles) - 1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ', 1)
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = library_
        except ValueError:
            # rsplitで分割できない場合（スペースが1つ、またはない場合）
            parts = eles[i].strip().split(' ', 1)
            if len(parts) == 2:
                module_, library_ = parts
            else:
                module_ = parts[0]
                library_ = module_ # 次のライブラリは同じモジュール名を引き継ぐ
            
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = library_

    api = first_lib.strip() + '.' + eles[-1].strip()
    api = api.lower().replace(' ', '')
    apis.append(api)
    return apis[0]

def calculate_head_or_tail(api_stats_list):
    """
    APIの登場回数に基づいてヘッド(上位50%)とテール(下位50%)を決定する。
    """
    total_appearance_count = sum(entry["first_appearance_count"] for entry in api_stats_list)
    
    if total_appearance_count == 0:
        return

    # 出現回数で降順ソート
    sorted_api_stats = sorted(api_stats_list, key=lambda x: x["first_appearance_count"], reverse=True)
    
    cumulative_count = 0
    head_threshold = total_appearance_count * 0.5  # 上位50%の閾値
    
    for entry in sorted_api_stats:
        cumulative_count += entry["first_appearance_count"]
        if cumulative_count <= head_threshold:
            entry["head_or_tail"] = 0  # ヘッド
        else:
            entry["head_or_tail"] = 1  # テール

def main():
    parser = argparse.ArgumentParser(description="""
        正解APIと各モデルの予測を比較し、統計情報を出力するスクリプト。
        各ファイルの先頭10万件を対象とします。
    """)
    # --- 引数の設定（分かりやすい名前に変更） ---
    parser.add_argument('--train_file', type=str, default='./data/train_3_lines.csv', 
                        help="正解ラベルが含まれる訓練データファイル(CSV)")
    parser.add_argument('--codebert_preds_file', type=str, default='./data/CodeBERT_predictions_train100000.txt',
                        help="CodeBERTの予測結果ファイル")
    parser.add_argument('--codet5_preds_file', type=str, default='./data/CodeT5_predictions_train100000.txt',
                        help="CodeT5の予測結果ファイル")
    parser.add_argument('--mularec_preds_file', type=str, default='./data/MulaRec_predictions_train100000.txt',
                        help="MulaRecの予測結果ファイル")
    parser.add_argument('--output_dir', type=str, default='./data/',
                        help="出力JSONファイルを保存するディレクトリ")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- データの読み込みと10万件への制限 ---
    limit = 100000
    print(f"各ファイルの先頭 {limit} 件を読み込んで処理します。")

    # 訓練データ（正解ラベル）の読み込み
    try:
        df_train = pd.read_csv(args.train_file)
        # 'target_api'カラムの存在を想定。もしカラム名が違う場合はここを修正してください。
        correct_refs = df_train['target_api'].head(limit).apply(parse_string_into_apis).tolist()
    except FileNotFoundError:
        print(f"エラー: 訓練ファイルが見つかりません: {args.train_file}")
        return
    except KeyError:
        print(f"エラー: 訓練ファイルに 'target_api' カラムが見つかりません。カラム名を確認してください。")
        return

    # 予測結果の読み込み
    with open(args.codebert_preds_file, 'r', encoding='utf-8') as f:
        codebert_preds = [parse_string_into_apis(pred) for pred in f.readlines()[:limit]]
    with open(args.codet5_preds_file, 'r', encoding='utf-8') as f:
        codet5_preds = [parse_string_into_apis(pred) for pred in f.readlines()[:limit]]
    with open(args.mularec_preds_file, 'r', encoding='utf-8') as f:
        mularec_preds = [parse_string_into_apis(pred) for pred in f.readlines()[:limit]]

    # 読み込んだデータ件数の整合性を確認
    num_records = min(len(correct_refs), len(codebert_preds), len(codet5_preds), len(mularec_preds))
    if num_records < limit:
        print(f"警告: ファイルのいずれかの行数が {limit} 未満です。利用可能な最小件数 {num_records} 件で処理を続行します。")

    # --- 統計情報の集計 ---
    api_stats = defaultdict(lambda: {
        "first_appearance_count": 0,
        "CodeBERT_first_appearance_count": 0, "CodeBERT_correct_count": 0,
        "CodeT5_first_appearance_count": 0, "CodeT5_correct_count": 0,
        "MulaRec_first_appearance_count": 0, "MulaRec_correct_count": 0,
    })

    # 各ファイルの先頭から順番に比較
    for i in range(num_records):
        correct_api = correct_refs[i]
        codebert_api = codebert_preds[i]
        codet5_api = codet5_preds[i]
        mularec_api = mularec_preds[i]

        # 正解APIメソッドの登場回数をカウント
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

    # --- 集計結果の整形と出力 ---
    api_stats_list = []
    for api_method, stats in api_stats.items():
        entry = {"api_method": api_method}
        entry.update(stats)
        api_stats_list.append(entry)

    # first_appearance_countで降順ソート
    api_stats_list = sorted(api_stats_list, key=lambda x: x["first_appearance_count"], reverse=True)

    # head_or_tail を計算して追加
    calculate_head_or_tail(api_stats_list)

    # ヘッドとテールの統計を計算
    head_entries = [e for e in api_stats_list if e.get("head_or_tail") == 0]
    tail_entries = [e for e in api_stats_list if e.get("head_or_tail") == 1]
    
    head_count = len(head_entries)
    tail_count = len(tail_entries)
    head_appearance_count = sum(e["first_appearance_count"] for e in head_entries)
    tail_appearance_count = sum(e["first_appearance_count"] for e in tail_entries)

    # 結果を標準出力
    print("-" * 30)
    print("集計結果:")
    print(f"ヘッドのAPIメソッド: {head_count} 種類, 合計 {head_appearance_count} 件")
    print(f"テールのAPIメソッド: {tail_count} 種類, 合計 {tail_appearance_count} 件")
    print("-" * 30)

    # JSONファイルに出力
    output_path = os.path.join(args.output_dir, 'first_correctCount_and_appearCount.json')
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(api_stats_list, outfile, indent=4, ensure_ascii=False)
    
    print(f"詳細な統計情報がJSONファイルに出力されました: {output_path}")


if __name__ == "__main__":
    main()