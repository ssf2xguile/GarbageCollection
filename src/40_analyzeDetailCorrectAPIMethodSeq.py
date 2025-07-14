import pandas as pd
import argparse
import json
import math

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('\t')[0].strip().split('.')
    first_lib = eles[0]

    for i in range(1, len(eles) - 1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ', 1)
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

def calculate_average_correct_achievement_rate(api_sequence, model_name, api_dict_local):
    """APIシーケンスの平均正解実績率を計算する"""
    accuracy = 1.0
    seq_len = len(api_sequence)
    for api in api_sequence:
        try:
            correct_count = api_dict_local[api][f"{model_name}_correct_count"]
            appearance_count = api_dict_local[api]["first_appearance_count"]
            if appearance_count == 0:
                accuracy *= 0  # 出現がない場合は正解率0とする
            else:
                accuracy *= (correct_count / appearance_count)
        except KeyError:
            accuracy *= 0  # APIが辞書に存在しない場合は正解率0
    accuracy = math.pow(accuracy, 1 / seq_len)  # 平均正解実績率を計算
    return accuracy

def calculate_head_api_ratio(api_sequence, api_dict_local):
    """APIシーケンス中のヘッドAPIメソッドの割合を計算する"""
    seq_len = len(api_sequence)
    if seq_len == 0:
        return 0.0

    head_api_count = 0
    for api in api_sequence:
        try:
            if api_dict_local[api]["head_or_tail"] == 0: # Assuming 0 means head
                head_api_count += 1
        except KeyError:
            pass # API not in dict, or "head_or_tail" key missing
    
    return head_api_count / seq_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file (used to determine data size if needed, though currently unused)")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to CodeBERT predictions file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to CodeT5 predictions file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to MulaRec predictions file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="Path to the API database JSON file")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test CSV file with correct targets")
    parser.add_argument('--csv_output_file', type=str, default='./data/rejected_but_correct_analysis.csv', help="Path to save the output CSV file")
    args = parser.parse_args()

    # Load model predictions
    with open(args.file2, 'r') as file2:
        codebert_model_predictions = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r') as file3:
        codet5_model_predictions = [parse_string_into_apis(pred) for pred in file3.readlines()]
    with open(args.file4, 'r') as file4:
        mularec_model_predictions = [parse_string_into_apis(pred) for pred in file4.readlines()]

    # Load ground truth data
    df_test = pd.read_csv(args.test_file)
    correct_refs = df_test['target_api'].apply(parse_string_into_apis).tolist()
    original_indices = df_test['api_index'].tolist() # Renamed to avoid conflict

    # Load API database with normalized keys
    with open(args.api_database, 'r', encoding='utf-8') as file:
        api_database_json = json.load(file)
    api_dict_global = {
        entry["api_method"].lower().replace(' ', ''): entry 
        for entry in api_database_json
    }

    # Map predictions based on original_index from the test file
    codebert_preds_mapped = []
    codet5_preds_mapped = []
    mularec_preds_mapped = []
    for index_val in original_indices:
        # Assuming original_index is 1-based
        codebert_preds_mapped.append(codebert_model_predictions[int(index_val)-1])
        codet5_preds_mapped.append(codet5_model_predictions[int(index_val)-1])
        mularec_preds_mapped.append(mularec_model_predictions[int(index_val)-1])
    
    # Determine final ensemble predictions
    final_apiseq_list = []
    for cb_pred, ct5_pred, mula_pred in zip(codebert_preds_mapped, codet5_preds_mapped, mularec_preds_mapped):
        if cb_pred == ct5_pred == mula_pred:
            final_apiseq_list.append(cb_pred)
        elif cb_pred == ct5_pred:
            final_apiseq_list.append(cb_pred)
        elif cb_pred == mula_pred:
            final_apiseq_list.append(cb_pred)
        elif ct5_pred == mula_pred:
            final_apiseq_list.append(ct5_pred)
        else: # All three models differ
            final_apiseq_list.append(['none.none']) # Rejected by simple ensemble

    csv_output_data = []
    num_testdata = len(correct_refs)

    for i in range(num_testdata):
        final_ensemble_prediction = final_apiseq_list[i]
        true_correct_sequence = correct_refs[i]
        
        current_cb_pred = codebert_preds_mapped[i]
        current_ct5_pred = codet5_preds_mapped[i]
        current_mula_pred = mularec_preds_mapped[i]

        if final_ensemble_prediction == ['none.none']: # If ensemble rejected
            is_cb_correct = (current_cb_pred == true_correct_sequence)
            is_ct5_correct = (current_ct5_pred == true_correct_sequence)
            is_mula_correct = (current_mula_pred == true_correct_sequence)

            if is_cb_correct or is_ct5_correct or is_mula_correct:
                # This rejected case had at least one model being correct
                correct_model_names = []
                if is_cb_correct: correct_model_names.append("CodeBERT")
                if is_ct5_correct: correct_model_names.append("CodeT5")
                if is_mula_correct: correct_model_names.append("MulaRec")

                cb_ach_rate = calculate_average_correct_achievement_rate(current_cb_pred, "CodeBERT", api_dict_global)
                ct5_ach_rate = calculate_average_correct_achievement_rate(current_ct5_pred, "CodeT5", api_dict_global)
                mula_ach_rate = calculate_average_correct_achievement_rate(current_mula_pred, "MulaRec", api_dict_global)

                cb_head_rate = calculate_head_api_ratio(current_cb_pred, api_dict_global)
                ct5_head_rate = calculate_head_api_ratio(current_ct5_pred, api_dict_global)
                mula_head_rate = calculate_head_api_ratio(current_mula_pred, api_dict_global)

                row_data = {
                    "CodeBERT_Prediction": str(current_cb_pred),
                    "CodeT5_Prediction": str(current_ct5_pred),
                    "MulaRec_Prediction": str(current_mula_pred),
                    "Correct_Model_Names": ", ".join(correct_model_names),
                    "Correct_API_Sequence": str(true_correct_sequence),
                    "CodeBERT_Achievement_Rate": cb_ach_rate,
                    "CodeT5_Achievement_Rate": ct5_ach_rate,
                    "MulaRec_Achievement_Rate": mula_ach_rate,
                    "CodeBERT_Head_API_Rate": cb_head_rate,
                    "CodeT5_Head_API_Rate": ct5_head_rate,
                    "MulaRec_Head_API_Rate": mula_head_rate,
                }
                csv_output_data.append(row_data)

    # Create DataFrame and save to CSV
    if csv_output_data:
        output_df = pd.DataFrame(csv_output_data)
        output_df.to_csv(args.csv_output_file, index=False, encoding='utf-8-sig')
        print(f"Analysis of rejected but correct sequences saved to: {args.csv_output_file}")
    else:
        print("No instances found where a rejected sequence had an underlying correct model prediction.")

if __name__ == '__main__':
    main()