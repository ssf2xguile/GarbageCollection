import pandas as pd
import argparse
import json

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

def classify_case(correct, final, preds):
    if final == ['none.none']:
        correct_matches = [pred == correct for pred in preds]
        if correct_matches.count(True) == 1:
            return 'C1'
        else:
            return 'C2'
    else:
        if final != correct:
            correct_matches = [pred == correct for pred in preds]
            if correct_matches.count(True) == 1:
                return 'E2'
            else:
                return 'E1'
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to save the output file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="whole")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    parser.add_argument('--output_file', type=str, default='./data/apiseq_recommendation_comparison_90.png', help="Path to save the output file")
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

    apiseq_accuracy_list = []
    apiseq_rejection_rate_list = []
    final_apiseq = []
    for codebert_pred, codet5_pred, mularec_pred in zip(codebert_preds, codet5_preds, mularec_preds):
        if codebert_pred == codet5_pred == mularec_pred:
            final_apiseq.append(codebert_pred)
        elif codebert_pred == codet5_pred:
            final_apiseq.append(codebert_pred)
        elif codebert_pred == mularec_pred:
            final_apiseq.append(codebert_pred)
        elif codet5_pred == mularec_pred:
            final_apiseq.append(codet5_pred)
        else:
            final_apiseq.append(['none.none'])

    accu_count = 0
    num_testdata = 0
    skip_count = 0
    for idx, api_seq, final_pred in zip(original_index, correct_refs, final_apiseq):
        if 'none.none' in final_pred:
            skip_count += 1
        else:
            if final_pred == api_seq:
                accu_count += 1
        num_testdata += 1
    apiseq_final_accuracy = round((accu_count / (18500 - skip_count))*100,2)
    apiseq_final_rejection_rate = round(((skip_count / 18500))*100,2)
    print("apiseq_final_accuracy: ", apiseq_final_accuracy)
    print("apiseq_final_rejection_rate: ", apiseq_final_rejection_rate)
    print("skip_count: ", skip_count)

    rejected_with_one_correct = 0       # 棄却されたが、1モデルは正解していた（惜しい棄却）
    rejected_all_incorrect = 0          # 棄却され、全モデルが不正解だった（妥当な棄却）
    wrong_output_all_incorrect = 0      # 出力したが、すべてのモデルが不正解（回避不能な不正解）
    wrong_output_with_one_correct = 0   # 出力したが、1モデルだけが正解していた（拾えなかった正解）

    for correct, final, cb, ct5, mula in zip(correct_refs, final_apiseq, codebert_preds, codet5_preds, mularec_preds):
        label = classify_case(correct, final, [cb, ct5, mula])
        if label == 'C1':
            rejected_with_one_correct += 1
        elif label == 'C2':
            rejected_all_incorrect += 1
        elif label == 'E1':
            wrong_output_all_incorrect += 1
        elif label == 'E2':
            wrong_output_with_one_correct += 1

    # --- 明確な出力名に変更 ---
    print("Rejected but one model was correct (rejected_with_one_correct):", rejected_with_one_correct)
    print("Rejected and all models were incorrect (rejected_all_incorrect):", rejected_all_incorrect)
    print("Output was wrong and all models were incorrect (wrong_output_all_incorrect):", wrong_output_all_incorrect)
    print("Output was wrong but one model was correct (wrong_output_with_one_correct):", wrong_output_with_one_correct)
if __name__ == '__main__':
    main()