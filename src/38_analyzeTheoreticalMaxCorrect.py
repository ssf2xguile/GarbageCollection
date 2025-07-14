import pandas as pd
import argparse
import json
import math # math.pow を使用するために追加

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

def calculate_sequence_accuracy(api_sequence, model_name, api_dict):
    accuracy = 1.0
    seq_len = len(api_sequence)
    for api in api_sequence:
        try:
            correct_count = api_dict[api][f"{model_name}_correct_count"]
            appearance_count = api_dict[api]["first_appearance_count"]
            if appearance_count == 0:
                accuracy *= 0  # 出現がない場合は正解率0とする
            else:
                accuracy *= (correct_count / appearance_count)
        except KeyError:
            accuracy *= 0  # APIが辞書に存在しない場合は正解率0
    accuracy = math.pow(accuracy, 1 / seq_len)  # 平均正解率を計算
    return accuracy

# APIシーケンス中のヘッドAPIメソッドの割合を計算する関数
def calculate_head_api_ratio(api_sequence, api_dict_local):
    seq_len = len(api_sequence)
    if seq_len == 0:
        return 0.0

    head_api_count = 0
    for api in api_sequence:
        try:
            # head_or_tail == 0 をヘッドAPIと仮定
            if api_dict_local[api]["head_or_tail"] == 0:
                head_api_count += 1
        except KeyError:
            # APIが辞書に存在しない場合は無視 (またはエラー処理)
            pass 
    
    return head_api_count / seq_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to save the output file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="Path to the api_model_correct_count.json like file")
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

    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    # APIデータベースを読み込む (api_dict_global とリネームして明確化)
    with open(args.api_database, 'r', encoding='utf-8') as file:
            api_database_json = json.load(file)
    api_dict_global = {entry["api_method"]: entry for entry in api_database_json}

    codebert_preds_mapped = []
    codet5_preds_mapped = []
    mularec_preds_mapped = []
    for index_val in original_index:
        codebert_preds_mapped.append(codebert_predictions[int(index_val)-1])
        codet5_preds_mapped.append(codet5_predictions[int(index_val)-1])
        mularec_preds_mapped.append(mularec_predictions[int(index_val)-1])

    final_apiseq = []
    for cb_pred, ct5_pred, mula_pred in zip(codebert_preds_mapped, codet5_preds_mapped, mularec_preds_mapped):
        if cb_pred == ct5_pred == mula_pred:
            final_apiseq.append(cb_pred)
        elif cb_pred == ct5_pred:
            final_apiseq.append(cb_pred)
        elif cb_pred == mula_pred:
            final_apiseq.append(cb_pred)
        elif ct5_pred == mula_pred:
            final_apiseq.append(ct5_pred)
        else:
            final_apiseq.append(['none.none'])

    accu_count = 0
    num_testdata = 0
    skip_count = 0
    for final_pred, api_seq in zip(final_apiseq, correct_refs):
        if 'none.none' in final_pred:
            skip_count += 1
        else:
            if final_pred == api_seq:
                accu_count += 1
        num_testdata += 1
    
    effective_test_data_count = num_testdata - skip_count
    if effective_test_data_count == 0:
        apiseq_final_accuracy = 0.0
    else:
        apiseq_final_accuracy = round((accu_count / effective_test_data_count)*100,2)
    
    if num_testdata == 0:
        apiseq_final_rejection_rate = 0.0
    else:
        apiseq_final_rejection_rate = round(((skip_count / num_testdata))*100,2)

    print("apiseq_final_accuracy: ", apiseq_final_accuracy)
    print("apiseq_final_rejection_rate: ", apiseq_final_rejection_rate)

    rejected_sequences_with_correct_model_details = []
    
    codebert_correct_in_rejected_count = 0
    codet5_correct_in_rejected_count = 0
    mularec_correct_in_rejected_count = 0

    # モデルごとの指標集計用辞書
    model_names = ["CodeBERT", "CodeT5", "MulaRec"]
    model_metrics_summary = {model: {"total_accuracy": 0.0, "total_head_ratio": 0.0, "count": 0} for model in model_names}

    for i, (orig_idx, correct_seq, final_pred_seq, cb_pred_seq, ct5_pred_seq, mula_pred_seq) in \
            enumerate(zip(original_index, correct_refs, final_apiseq, 
                          codebert_preds_mapped, codet5_preds_mapped, mularec_preds_mapped)):
        
        if final_pred_seq == ['none.none']: 
            is_cb_correct = (cb_pred_seq == correct_seq)
            is_ct5_correct = (ct5_pred_seq == correct_seq)
            is_mula_correct = (mula_pred_seq == correct_seq)
            
            current_correct_models = []
            if is_cb_correct: current_correct_models.append("CodeBERT")
            if is_ct5_correct: current_correct_models.append("CodeT5")
            if is_mula_correct: current_correct_models.append("MulaRec")
            
            if current_correct_models:
                details = {
                    "original_index": orig_idx,
                    "correct_models": current_correct_models,
                    "correct_sequence": correct_seq
                }
                rejected_sequences_with_correct_model_details.append(details)

                if is_cb_correct:
                    codebert_correct_in_rejected_count += 1
                    seq_acc = calculate_sequence_accuracy(correct_seq, "CodeBERT", api_dict_global)
                    head_ratio = calculate_head_api_ratio(correct_seq, api_dict_global)
                    model_metrics_summary["CodeBERT"]["total_accuracy"] += seq_acc
                    model_metrics_summary["CodeBERT"]["total_head_ratio"] += head_ratio
                    model_metrics_summary["CodeBERT"]["count"] += 1
                if is_ct5_correct:
                    codet5_correct_in_rejected_count += 1
                    seq_acc = calculate_sequence_accuracy(correct_seq, "CodeT5", api_dict_global)
                    head_ratio = calculate_head_api_ratio(correct_seq, api_dict_global)
                    model_metrics_summary["CodeT5"]["total_accuracy"] += seq_acc
                    model_metrics_summary["CodeT5"]["total_head_ratio"] += head_ratio
                    model_metrics_summary["CodeT5"]["count"] += 1
                if is_mula_correct:
                    mularec_correct_in_rejected_count += 1
                    seq_acc = calculate_sequence_accuracy(correct_seq, "MulaRec", api_dict_global)
                    head_ratio = calculate_head_api_ratio(correct_seq, api_dict_global)
                    model_metrics_summary["MulaRec"]["total_accuracy"] += seq_acc
                    model_metrics_summary["MulaRec"]["total_head_ratio"] += head_ratio
                    model_metrics_summary["MulaRec"]["count"] += 1
                    
    print(f"棄却された総件数: {skip_count}")
    print(f"棄却されたAPIメソッドシーケンスのうち、どれか1つでも正解していた件数: {len(rejected_sequences_with_correct_model_details)}")
    """
    if rejected_sequences_with_correct_model_details:
        print("\n棄却されたが、いずれかのモデルが正解していたAPIシーケンスの詳細 (インデックス: モデル名 - APIシーケンス):")
        for detail in rejected_sequences_with_correct_model_details:
            for model_name in detail["correct_models"]:
                api_sequence_str = str(detail["correct_sequence"])
                print(f"  インデックス {detail['original_index']}: {model_name} - {api_sequence_str}")
    """
    print("\n--- まとめ: 棄却されたシーケンスのうち、各モデルが正解を出力していた件数 ---")
    print(f"CodeBERT: {codebert_correct_in_rejected_count}件")
    print(f"CodeT5: {codet5_correct_in_rejected_count}件")
    print(f"MulaRec: {mularec_correct_in_rejected_count}件")

    print("\n--- まとめ: 棄却されたシーケンスで各モデルが正解した場合の追加指標 ---")
    for model_name in model_names:
        count = model_metrics_summary[model_name]["count"]
        if count > 0:
            avg_accuracy = model_metrics_summary[model_name]["total_accuracy"] / count
            avg_head_ratio = model_metrics_summary[model_name]["total_head_ratio"] / count
            print(f"{model_name}:")
            print(f"平均正解率 (ユーザー定義): {avg_accuracy:.4f}")
            print(f"平均ヘッドAPIメソッド割合: {avg_head_ratio:.4f}")
        else:
            print(f"{model_name}: (該当する正解シーケンスなし)")

if __name__ == '__main__':
    main()