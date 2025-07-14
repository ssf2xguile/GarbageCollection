"""
調査用データに対して各モデルのAPIが何種類何回登場したのかをカウントするプログラム
入力: 各モデルが予測したAPIシーケンスのファイル(CodeBERT_predictions.txt, CodeT5_predictions.txt, MulaRec_predictions.txt)と、元のテストデータのインデックスがついている調査用データ
出力: jsonファイル
"""
import pandas as pd
import argparse
import json
import os

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

def recreate_apimethod_list(model_preds, original_index):
    relevant_apimethod = []
    for index in original_index:
        relevant_apimethod.append(model_preds[int(index)-1])
    return relevant_apimethod

def aggregate_model_counts(codebert_apis, codet5_apis, mularec_apis):
    all_apis = set(codebert_apis + codet5_apis + mularec_apis)
    aggregated_data = []

    for api in all_apis:
        codebert_count = codebert_apis.count(api)
        codet5_count = codet5_apis.count(api)
        mularec_count = mularec_apis.count(api)

        aggregated_data.append({
            "api_method": api,
            "CodeBERT_first_appearance_count": codebert_count,
            "CodeT5_first_appearance_count": codet5_count,
            "MulaRec_first_appearance_count": mularec_count
        })

    return aggregated_data

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
        codebert_preds_file = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds_file = [parse_string_into_apis(pred) for pred in file2.readlines()]    # [...,'base64.decodetoobject, 'math.sqrt', 'thread.start',...]のようになる 
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds_file = [parse_string_into_apis(pred) for pred in file3.readlines()]   # [...,'base64.decodetoobject', 'point.math', 'thread.start',...]のようになる

    # 調査対象データに該当する予測結果のインデックスのリストを取り出す
    df = pd.read_csv(args.investigate_file)
    original_index = df['api_index'].tolist()   # 元のテストデータの何番目のデータかを示すインデックス[..., 36149, 36154, ...]のようになる 
    
    # 各モデルの正解数をカウントする
    codebert_apis = recreate_apimethod_list(codebert_preds_file, original_index)
    codet5_apis = recreate_apimethod_list(codet5_preds_file, original_index)
    mularec_apis = recreate_apimethod_list(mularec_preds_file, original_index)

    # APIメソッドごとに各モデルの予測回数を集約
    aggregated_data = aggregate_model_counts(codebert_apis, codet5_apis, mularec_apis)

    # 集約結果をJSONファイルに保存
    with open(os.path.join(args.output_dir, 'first_prediction_api.json'), 'w', encoding='utf-8') as outfile:
        json.dump(aggregated_data, outfile, ensure_ascii=False, indent=4)

    print("Aggregated data has been saved successfully.")


if __name__ == "__main__":
    main()