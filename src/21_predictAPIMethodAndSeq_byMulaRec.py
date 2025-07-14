# CodeT5による入力のテール判定→ヘッドインデックス取得→テストデータと合致するインデックスのみフィルタリング→MulaRecによる単一APIメソッド推薦&APIメソッドシーケンス推薦→正解数算出。
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
    return apis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeT5_tail_prediction.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    args = parser.parse_args()

    with open(args.file1, 'r') as f:
        codet5_taiL_prediction = [int(line.strip()) for line in f if line.strip()]
    with open(args.file2, 'r') as file2:
        mularec_api_predictions = [parse_string_into_apis(pred) for pred in file2.readlines()]

    # ヘッドデータ (0) のインデックスを取得
    head_indices = [i for i, value in enumerate(codet5_taiL_prediction) if value == 0]
    print('Head indices count: ', len(head_indices))    

    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    accu_api_seq_index = []
    accu_api_method_index = []
    numHeadInput_in_testdata = 0
    # テストデータと合致するインデックスのみフィルタリング
    for idx, api_seq in zip(original_index, correct_refs):
        if codet5_taiL_prediction[int(idx)-1] == 0:
            if mularec_api_predictions[int(idx)-1] == api_seq:
                accu_api_seq_index.append(int(idx)-1)
            if mularec_api_predictions[int(idx)-1][0] == api_seq[0]:
                accu_api_method_index.append(int(idx)-1)
            numHeadInput_in_testdata += 1
    print('Head Input in Test Data: ', numHeadInput_in_testdata)
    print('API Sequence Accurate Count: ', len(accu_api_seq_index))
    print('API Method Accurate Count: ', len(accu_api_method_index))

if __name__ == '__main__':
    main()