"""訓練データ(10万件)の予測精度をAPIメソッドシーケンス単位で評価するスクリプト。"""
import pandas as pd
import argparse
import os

def parse_string_into_apis(str_):
    """
    タブ区切りの予測文字列からAPIシーケンス（リスト）を抽出する。
    """
    if not isinstance(str_, str):
        return [] # 空のリストを返す
        
    apis = []
    prediction_part = str_.split('\t')[0]
    eles = prediction_part.strip().split('.')

    if len(eles) < 2:
        # パース不能な場合は、元の文字列を要素とするリストを返す
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
    
    # ★★★ 修正点: 最初のAPIメソッドではなく、APIシーケンスのリスト全体を返す ★★★
    return apis

def calculate_accuracy(model_name, predictions, ground_truths):
    """
    特定のモデルの正解数と正解率を計算して表示する。
    比較はAPIシーケンス（リスト）単位で行う。
    """
    num_to_compare = min(len(predictions), len(ground_truths))
    
    correct_count = 0
    for i in range(num_to_compare):
        # リスト同士を直接比較することで、シーケンス全体の一致を確認
        if predictions[i] == ground_truths[i]:
            correct_count += 1
            
    accuracy = (correct_count / num_to_compare) * 100 if num_to_compare > 0 else 0
    
    print(f"--- {model_name} の評価結果 (APIメソッドシーケンス単位) ---")
    print(f"  正解数: {correct_count} / {num_to_compare}")
    print(f"  正解率: {accuracy:.2f}%")
    print("-" * (len(model_name) + 25))

def main():
    parser = argparse.ArgumentParser(description="""
        各モデルの予測精度をAPIシーケンス単位で評価するスクリプト。
        各ファイルの先頭10万件を対象とします。
    """)
    parser.add_argument('--train_file', type=str, default='./data/train_3_lines.csv', 
                        help="正解ラベルが含まれる訓練データファイル(CSV)")
    parser.add_argument('--codebert_preds_file', type=str, default='./data/CodeBERT_predictions_train100000.txt',
                        help="CodeBERTの予測結果ファイル")
    parser.add_argument('--codet5_preds_file', type=str, default='./data/CodeT5_predictions_train100000.txt',
                        help="CodeT5の予測結果ファイル")
    parser.add_argument('--mularec_preds_file', type=str, default='./data/MulaRec_predictions_train100000.txt',
                        help="MulaRecの予測結果ファイル")
    args = parser.parse_args()
    
    limit = 100000
    print(f"評価件数: {limit}件\n")

    # --- データの読み込み ---
    try:
        df_train = pd.read_csv(args.train_file)
        ground_truths = df_train['target_api'].head(limit).apply(parse_string_into_apis).tolist()
    except Exception as e:
        print(f"エラー: 正解データファイル '{args.train_file}' の読み込みに失敗しました。")
        print(e)
        return

    try:
        with open(args.codebert_preds_file, 'r', encoding='utf-8') as f:
            codebert_preds = [parse_string_into_apis(pred) for pred in f.readlines()[:limit]]
        with open(args.codet5_preds_file, 'r', encoding='utf-8') as f:
            codet5_preds = [parse_string_into_apis(pred) for pred in f.readlines()[:limit]]
        with open(args.mularec_preds_file, 'r', encoding='utf-8') as f:
            mularec_preds = [parse_string_into_apis(pred) for pred in f.readlines()[:limit]]
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