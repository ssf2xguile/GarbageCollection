# 3つのLLMによるExactmatchしたテストデータのうち、テールデータを含むテストデータのインデックスを洗い出す
import pandas as pd
import argparse
import json
import os

def update_model_counts(model_preds, correct_refs, api_database, model_name):
    """
    各モデルの予測結果に基づいて、正解のAPIメソッドと比較し、正解カウントを更新する関数。
    モデルの予測シーケンスが正解シーケンスより短い場合は予測シーケンスの長さで打ち切る。一方、予測シーケンスの方が長い場合は正解シーケンスの長さで比較を打ち切る。
    """
    for pred_seq, ref_seq in zip(model_preds, correct_refs):
        try:
            # 予測シーケンスと正解シーケンスを1つずつ比較
            for i in range(len(ref_seq)):
                pred_api = pred_seq[i]
                ref_api = ref_seq[i]  # 正解APIシーケンスの範囲外になった場合はIndexErrorが発生
                if pred_api == ref_api:
                    api_database[ref_api][f"{model_name}_correct_count"] += 1
        except IndexError:
            # 予測シーケンスが正解シーケンスより短くて範囲外になったら比較を打ち切る
            continue

def parse_string_into_apis(str_):   # 今回は小文字化と空白削除の処理を追加している。
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
    return apis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_predictions.txt', help="Path to save the output file")
    parser.add_argument('--test_file', type=str, default='./data/test_3_lines.csv', help="Path to save the output file")
    parser.add_argument('--head_or_tail_database', type=str, default='./data/api_head_or_tail.json', help="Path to save the output file")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the text file which including tail data")
    args = parser.parse_args()

    # 予測結果を読み込む
    with open(args.file1, 'r', encoding='utf-8') as file1:
        codebert_preds_file = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds_file = [parse_string_into_apis(pred) for pred in file2.readlines()]    # [['base64.decodetoobject'], ['math.sqrt', 'point.calculatedistance', 'math.sqrt'], ['thread.start'],...]のようになる 
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds_file = [parse_string_into_apis(pred) for pred in file3.readlines()]   # [['base64.decodetoobject'], ['point.math', 'math.abs', 'point.distance'], ['thread.start'],...]のようになる

    # JSONデータベースを読み込む
    with open(args.head_or_tail_database, 'r', encoding='utf-8') as file:
        api_database = json.load(file)

    # 各APIについて正解回数を初期化
    api_results = {}
    for entry in api_database:
        api_method = entry['api_method']
        appearance_count = entry.get('appearance_count', 0) # appearance_countを取得、デフォルトは0（未指定の場合）
        head_or_tail = entry.get('head_or_tail', -1)  # head_or_tailを取得、デフォルトは-1（未指定の場合）
        api_results[api_method] = {
            "api_method": api_method,
            "appearance_count": appearance_count,
            "head_or_tail": head_or_tail,
            "CodeBERT_correct_count": 0,
            "CodeT5_correct_count": 0,
            "MulaRec_correct_count": 0
        }

    # 正解データを加工する
    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()  # この処理によってリストのリストが得られる[['base64.decodetoobject'], ['math.sqrt', 'point.calculatedistance', 'math.sqrt'], ['thread.start'],...]
    
    # 各モデルの正解数をカウントする
    update_model_counts(codebert_preds_file, correct_refs, api_results, "CodeBERT")
    update_model_counts(codet5_preds_file, correct_refs, api_results, "CodeT5")
    update_model_counts(mularec_preds_file, correct_refs, api_results, "MulaRec")

    # 最終結果の出力
    output_file = os.path.join(args.output_dir, 'api_model_correct_count.json')
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(list(api_results.values()), json_file, indent=4)

    print(f"API correct count data has been saved to {output_file}")

if __name__ == "__main__":
    main()