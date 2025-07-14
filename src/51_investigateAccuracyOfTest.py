import pandas as pd
import argparse
import os

def parse_string_into_apis(str_):
    """
    タブ区切りの予測文字列からAPIシーケンス（リスト）を抽出する。
    """
    if not isinstance(str_, str):
        return []
        
    apis = []
    prediction_part = str_.split('\t')[0]
    eles = prediction_part.strip().split('.')

    if len(eles) < 2:
        return [prediction_part.strip().lower().replace(' ', '')]

    first_lib = eles[0]

    for i in range(1, len(eles) - 1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ', 1)
            api = f"{first_lib.strip()}.{module_.strip()}"
            apis.append(api.lower().replace(' ', ''))
            first_lib = library_
        except ValueError:
            parts = eles[i].strip().split(' ', 1)
            module_ = parts[0]
            library_ = parts[1] if len(parts) > 1 else module_
            
            api = f"{first_lib.strip()}.{module_.strip()}"
            apis.append(api.lower().replace(' ', ''))
            first_lib = library_

    api = f"{first_lib.strip()}.{eles[-1].strip()}"
    apis.append(api.lower().replace(' ', ''))
    
    return apis

def calculate_accuracy(model_name, predictions, ground_truths):
    """
    特定のモデルの正解数と正解率を計算して表示する。
    """
    num_to_compare = min(len(predictions), len(ground_truths))
    
    correct_count = 0
    for i in range(num_to_compare):
        if predictions[i] == ground_truths[i]:
            correct_count += 1
            
    accuracy = (correct_count / num_to_compare) * 100 if num_to_compare > 0 else 0
    
    print(f"--- {model_name} の評価結果 (APIメソッドシーケンス単位) ---")
    print(f"  評価件数: {num_to_compare}件")
    print(f"  正解数: {correct_count}件")
    print(f"  正解率: {accuracy:.2f}%")
    print("-" * (len(model_name) + 25))

def main():
    parser = argparse.ArgumentParser(description="""
        各モデルの予測精度をAPIシーケンス単位で評価するスクリプト（テストデータ版）。
    """)
    # --- 引数のデフォルト値をテストファイルに変更 ---
    parser.add_argument('--test_file', type=str, default='./data/test_3_lines.csv', 
                        help="正解ラベルが含まれるテストデータファイル(CSV)")
    parser.add_argument('--codebert_preds_file', type=str, default='./data/CodeBERT_predictions.txt',
                        help="CodeBERTの予測結果ファイル")
    parser.add_argument('--codet5_preds_file', type=str, default='./data/CodeT5_predictions.txt',
                        help="CodeT5の予測結果ファイル")
    parser.add_argument('--mularec_preds_file', type=str, default='./data/MulaRec_predictions.txt',
                        help="MulaRecの予測結果ファイル")
    args = parser.parse_args()
    
    print("ファイルを全件読み込んで評価します。\n")

    # --- データの読み込み（読み込み制限を撤廃） ---
    try:
        df_test = pd.read_csv(args.test_file)
        ground_truths = df_test['target_api'].apply(parse_string_into_apis).tolist()
    except Exception as e:
        print(f"エラー: 正解データファイル '{args.test_file}' の読み込みに失敗しました。")
        print(e)
        return

    try:
        with open(args.codebert_preds_file, 'r', encoding='utf-8') as f:
            codebert_preds = [parse_string_into_apis(pred) for pred in f.readlines()]
        with open(args.codet5_preds_file, 'r', encoding='utf-8') as f:
            codet5_preds = [parse_string_into_apis(pred) for pred in f.readlines()]
        with open(args.mularec_preds_file, 'r', encoding='utf-8') as f:
            mularec_preds = [parse_string_into_apis(pred) for pred in f.readlines()]
    except Exception as e:
        print(f"エラー: 予測結果ファイルの読み込み中に問題が発生しました。")
        print(e)
        return

    # --- 各モデルの正解率を計算して出力 ---
    calculate_accuracy("CodeBERT", codebert_preds, ground_truths)
    calculate_accuracy("CodeT5", codet5_preds, ground_truths)
    calculate_accuracy("MulaRec", mularec_preds, ground_truths)

if __name__ == "__main__":
    main()