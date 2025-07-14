# 3つのLLMによるExactmatchしたテストデータのうち、テールデータを含むテストデータのインデックスを洗い出す
import pandas as pd
import argparse
import json
import os

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
    return apis[0]

def check_api_registration_in_database(model_pred, api_dict, model_name):
    if model_pred not in api_dict:
        return 'none.none'
    elif api_dict[model_pred][f"{model_name}_first_appearance_count"] == 0:
        return 'none.none'
    else:
        return model_pred
    

def caluculate_final_accuracy(final_preds, correct_refs):
    correct_count = 0
    skip_count = 0
    skip_index_list = []
    for skip_index, (pred, ref) in enumerate(zip(final_preds, correct_refs)):
        if pred == 'none.none':
            skip_count += 1
            skip_index_list.append(skip_index)
        elif pred == ref:
            correct_count += 1
    return round(correct_count / (len(correct_refs)-skip_count) * 100, 2), correct_count, skip_count

def caluculate_each_model_final_accuracy(model_preds, correct_refs):
    correct_count = 0
    for pred, ref in zip(model_preds, correct_refs):
        if pred == ref:
            correct_count += 1
    return round(correct_count / len(correct_refs) * 100, 2), correct_count


def get_plausible_apimethod(codebert_preds, codet5_preds, mularec_preds, api_dict):
    try:
        codebert_accuracy = api_dict[codebert_preds]["CodeBERT_correct_count"] / api_dict[codebert_preds]["first_appearance_count"]
    except ZeroDivisionError:
        codebert_accuracy = 0.0
    except KeyError:
        codebert_accuracy = 0.0
    try:
        codet5_accuracy = api_dict[codet5_preds]["CodeT5_correct_count"] / api_dict[codet5_preds]["first_appearance_count"]
    except ZeroDivisionError:
        codet5_accuracy = 0.0
    except KeyError:
        codet5_accuracy = 0.0
    try:
        mularec_accuracy = api_dict[mularec_preds]["MulaRec_correct_count"] / api_dict[mularec_preds]["first_appearance_count"]
    except ZeroDivisionError:
        mularec_accuracy = 0.0
    except KeyError:
        mularec_accuracy = 0.0
    accuracies = {
        "CodeBERT": codebert_accuracy,
        "CodeT5": codet5_accuracy,
        "MulaRec": mularec_accuracy
    }

    plausible_apimethod = None
    # 最も高い正解率を持つモデル名を取得
    highest_model = max(accuracies, key=accuracies.get)
    highest_accuracy = accuracies[highest_model]
    if highest_model == "CodeBERT":
        plausible_apimethod = codebert_preds
    elif highest_model == "CodeT5":
        plausible_apimethod = codet5_preds
    else:
        plausible_apimethod = mularec_preds
    if highest_accuracy == 0.0:
        plausible_apimethod = 'none.none'
    return plausible_apimethod


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_predictions.txt', help="Path to save the output file")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to save the output file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="whole")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the text file which including tail data")
    args = parser.parse_args()

    # 予測結果を読み込む
    with open(args.file1, 'r', encoding='utf-8') as file1:
        codebert_preds_file = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds_file = [parse_string_into_apis(pred) for pred in file2.readlines()]    # [...,'base64.decodetoobject, 'math.sqrt', 'thread.start',...]のようになる 
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds_file = [parse_string_into_apis(pred) for pred in file3.readlines()]   # [...,'base64.decodetoobject', 'point.math', 'thread.start',...]のようになる

    # 正解データを加工する
    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()  # この処理によってリストのリストが得られる[...,'base64.decodetoobject', 'math.sqrt', 'thread.start',...]
    original_index = df['api_index'].tolist()   # 元のテストデータの何番目のデータかを示すインデックス[..., 36149, 36154, ...]のようになる

    # APIデータベースを読み込む
    with open(args.api_database, 'r', encoding='utf-8') as file:
            api_database = json.load(file)
    # APIメソッド名をキー、詳細を値とする辞書に変換
    api_dict = {entry["api_method"]: entry for entry in api_database}

    codebert_preds = []
    codet5_preds = []
    mularec_preds = []
    for index in original_index:
        codebert_preds.append(codebert_preds_file[int(index)-1])
        codet5_preds.append(codet5_preds_file[int(index)-1])
        mularec_preds.append(mularec_preds_file[int(index)-1])
    
    final_preds = []
    for codebert_pred, codet5_pred, mularec_pred in zip(codebert_preds, codet5_preds, mularec_preds):
        codebert_pred = check_api_registration_in_database(codebert_pred, api_dict, "CodeBERT")
        codet5_pred = check_api_registration_in_database(codet5_pred, api_dict, "CodeT5")
        mularec_pred = check_api_registration_in_database(mularec_pred, api_dict, "MulaRec")
        if codebert_pred == 'none.none' or codet5_pred == 'none.none' or mularec_pred == 'none.none':
            final_preds.append('none.none')
        else:   # 3つの予測結果がすべて 'none.none' でない場合
            if codebert_pred == codet5_pred == mularec_pred:
                final_preds.append(codebert_pred)
            elif codebert_pred == codet5_pred:
                final_preds.append(mularec_pred)
            elif codebert_pred == mularec_pred:
                final_preds.append(mularec_pred)
            elif codet5_pred == mularec_pred:
                final_preds.append(mularec_pred)
            else:
                plausible_apimethod = get_plausible_apimethod(codebert_pred, codet5_pred, mularec_pred, api_dict)
                final_preds.append(plausible_apimethod)

    # 予測結果の正解率を計算
    final_accuracy, correct_count, skip_count = caluculate_final_accuracy(final_preds, correct_refs)
    print(f"Final Accuracy: {final_accuracy}%")
    print(f"{correct_count}/{len(correct_refs)-skip_count} were correct.")

    # 各モデルの正解率を計算
    codebert_final_accuracy, codebert_correct_count = caluculate_each_model_final_accuracy(codebert_preds, correct_refs)
    codet5_final_accuracy, codet5_correct_count = caluculate_each_model_final_accuracy(codet5_preds, correct_refs)
    mularec_final_accuracy, mularec_correct_count = caluculate_each_model_final_accuracy(mularec_preds, correct_refs)
    print(f"CodeBERT Accuracy: {codebert_final_accuracy}%")
    print(f"{codebert_correct_count}/{len(correct_refs)} were correct.")
    print(f"CodeT5 Accuracy: {codet5_final_accuracy}%")
    print(f"{codet5_correct_count}/{len(correct_refs)} were correct.")
    print(f"MulaRec Accuracy: {mularec_final_accuracy}%")
    print(f"{mularec_correct_count}/{len(correct_refs)} were correct.")


if __name__ == "__main__":
    main()