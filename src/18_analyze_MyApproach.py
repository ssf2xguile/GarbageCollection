## 16_test_ecactmatch_withMyApproach.pyのコードをベースにして、提案手法の一部を分析する
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
    

def get_final_index(final_preds, correct_refs):
    unanswered_index_list = []
    correct_index_list = []
    wrong_index_list = []
    for index, (pred, ref) in enumerate(zip(final_preds, correct_refs)):
        if pred == 'none.none':
            unanswered_index_list.append(index)
        elif pred == ref:
            correct_index_list.append(index)
        else:
            wrong_index_list.append(index)
    return unanswered_index_list, correct_index_list, wrong_index_list

def unanswered_correct_wrong_list(final_preds, correct_refs):
    unanswered_correct_wrong = []
    for pred, ref in zip(final_preds, correct_refs):
        if pred == 'none.none':
            unanswered_correct_wrong.append(2)  # 未回答の場合
        elif pred == ref:
            unanswered_correct_wrong.append(1)  # 正解の場合
        else:
            unanswered_correct_wrong.append(0)  # 不正解の場合
    return unanswered_correct_wrong

def correct_wrong_list(model_preds, correct_refs):
    correct_wrong = []
    for pred, ref in zip(model_preds, correct_refs):
        if pred == ref:
            correct_wrong.append(1)  # 正解の場合
        else:
            correct_wrong.append(0)  # 不正解の場合
    return correct_wrong

def get_each_original_index(unanswered_index_list, correct_index_list, wrong_index_list, original_index):
    unanswered_original_index = [original_index[index] for index in unanswered_index_list]
    correct_original_index = [original_index[index] for index in correct_index_list]
    wrong_original_index = [original_index[index] for index in wrong_index_list]
    return unanswered_original_index, correct_original_index, wrong_original_index


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
                final_preds.append(codebert_pred)
            elif codebert_pred == mularec_pred:
                final_preds.append(codebert_pred)
            elif codet5_pred == mularec_pred:
                final_preds.append(codet5_pred)
            else:
                plausible_apimethod = get_plausible_apimethod(codebert_pred, codet5_pred, mularec_pred, api_dict)
                final_preds.append(plausible_apimethod)

    # 未回答、正解、不正解のインデックスを取得
    unanswered_index_list, correct_index_list, wrong_index_list = get_final_index(final_preds, correct_refs)
    final_status = unanswered_correct_wrong_list(final_preds, correct_refs)
    unanswered_original_index_list, correct_original_index_list, wrong_original_index_list = get_each_original_index(unanswered_index_list, correct_index_list, wrong_index_list, original_index)

    codebert_status = correct_wrong_list(codebert_preds, correct_refs)
    codet5_status = correct_wrong_list(codet5_preds, correct_refs)
    mularec_status = correct_wrong_list(mularec_preds, correct_refs)

    # CSVデータを準備
    csv_data = {
        'original_index': original_index,
        'final_status': final_status,
        'codebert_status': codebert_status,
        'codet5_status': codet5_status,
        'mularec_status': mularec_status,
        'correct_api': correct_refs,
        'final_api': final_preds,
        'codebert_api': codebert_preds,
        'codet5_api': codet5_preds,
        'mularec_api': mularec_preds
    }
    output_df = pd.DataFrame(csv_data)

    # original_indexを基準に昇順にソート
    output_df = output_df.sort_values(by='original_index')

    # CSVに書き出す
    output_df.to_csv(os.path.join(args.output_dir, 'MyApproachCorrectness.csv'), index=False)

if __name__ == "__main__":
    main()