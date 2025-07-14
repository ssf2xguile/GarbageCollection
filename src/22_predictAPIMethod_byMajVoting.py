# CodeT5による入力のテール判定→テールインデックス取得→テストデータと合致するインデックスのみフィルタリング→出力決定ルールによる単一APIメソッド推薦→正解数算出。
import pandas as pd
import argparse
import json
import os

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

def check_api_registration_in_database(model_pred, api_dict, model_name):
    if model_pred not in api_dict:
        return 'none.none'
    elif api_dict[model_pred][f"{model_name}_first_appearance_count"] == 0:
        return 'none.none'
    else:
        return model_pred

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
    parser.add_argument('--file1', type=str, default='./data/CodeT5_tail_prediction.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to save the output file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="whole")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    args = parser.parse_args()

    with open(args.file1, 'r') as f:
        codet5_taiL_prediction = [int(line.strip()) for line in f if line.strip()]
    with open(args.file2, 'r') as file2:
        codebert_predictions = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r') as file3:
        codet5_predictions = [parse_string_into_apis(pred) for pred in file3.readlines()]
    with open(args.file4, 'r') as file4:
        mularec_predictions = [parse_string_into_apis(pred) for pred in file4.readlines()]

    # ヘッドデータ (0) のインデックスを取得
    tail_indices = [i for i, value in enumerate(codet5_taiL_prediction) if value == 1]
    print('Tail indices count: ', len(tail_indices))    

    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    # APIデータベースを読み込む
    with open(args.api_database, 'r', encoding='utf-8') as file:
            api_database = json.load(file)
    # APIメソッド名をキー、詳細を値とする辞書に変換
    api_dict = {entry["api_method"]: entry for entry in api_database}

    codebert_preds = []
    codet5_preds = []
    mularec_preds = []
    for index in original_index:
        codebert_preds.append(codebert_predictions[int(index)-1])
        codet5_preds.append(codet5_predictions[int(index)-1])
        mularec_preds.append(mularec_predictions[int(index)-1])

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

    print(len(final_preds))
    accu_api_seq_index = []
    accu_api_method_index = []
    numTailInput_in_testdata = 0
    # テストデータと合致するインデックスのみフィルタリング
    for idx, api_seq, final_pred in zip(original_index, correct_refs, final_preds):
        if codet5_taiL_prediction[int(idx)-1] == 1:
            if final_pred == api_seq:
                accu_api_method_index.append(int(idx)-1)
            numTailInput_in_testdata += 1
    print('Tail Input in Test Data: ', numTailInput_in_testdata)
    print('API Sequence Accurate Count: ', len(accu_api_seq_index))
    print('API Method Accurate Count: ', len(accu_api_method_index))

if __name__ == '__main__':
    main()