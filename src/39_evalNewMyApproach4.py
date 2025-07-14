import pandas as pd
import argparse
import json
import math

def parse_string_into_apis(str_):
    apis = []
    # タブで区切られた最初の要素を取得し、ドットで分割
    eles = str_.split('\t')[0].strip().split('.')
    first_lib = eles[0]

    for i in range(1, len(eles) - 1):
        try:
            # 後ろからスペースで分割 (例: "module library")
            module_, library_ = eles[i].strip().rsplit(' ', 1)
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = library_
        except ValueError:
            try:
                # 前からスペースで分割 (例: "module library part2")
                module_, library_ = eles[i].strip().split(' ', 1)
                api = first_lib.strip() + '.' + module_.strip()
                api = api.lower().replace(' ', '')
                apis.append(api)
                first_lib = library_
            except ValueError:
                # スペースなし (例: "module")
                module_ = eles[i].strip()
                library_ = '' # 次の first_lib はこのモジュール名自体になる
                api = first_lib.strip() + '.' + module_.strip()
                api = api.lower().replace(' ', '')
                apis.append(api)
                first_lib = module_ # ライブラリがなければモジュール自体が次の基準

    # 最後の要素を追加
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
            # head_or_tail == 0 をヘッドAPIと仮定
            if api_dict_local[api]["head_or_tail"] == 0:
                head_api_count += 1
        except KeyError:
            # APIが辞書に存在しない場合はカウントしない
            pass
    
    return head_api_count / seq_len

def get_recommendation_by_new_rules(cb_pred, ct5_pred, mula_pred, api_dict_local):
    """新しいルールに基づいて推薦APIシーケンスを決定する"""
    # しきい値
    threshold_achievement_rate = 0.5
    threshold_head_ratio = 0.6

    models_metrics = {}
    all_models_passed_criteria = True # Flag to track if all models meet the criteria

    # Calculate metrics for all models
    for model_name, pred_sequence in [("MulaRec", mula_pred), ("CodeBERT", cb_pred), ("CodeT5", ct5_pred)]:
        achievement_rate = calculate_average_correct_achievement_rate(pred_sequence, model_name, api_dict_local)
        head_ratio = calculate_head_api_ratio(pred_sequence, api_dict_local)
        
        models_metrics[model_name] = {
            "achievement_rate": achievement_rate,
            "head_ratio": head_ratio,
            "sequence": pred_sequence
        }
        # Check if this model fails the criteria
        if not (achievement_rate >= threshold_achievement_rate and head_ratio >= threshold_head_ratio):
            all_models_passed_criteria = False
            # No need to store metrics if we already know not all models will pass,
            # but calculating all helps if debugging or if rules change later.
            # For this specific rule, we can break early if one fails and we only care about "all pass"
            # However, current user request implies checking all then deciding.
            # Re-evaluating: the request is "全てのモデルが両方のしきい値を超えた場合にMulaRecの結果を出力"
            # So if one fails, the overall condition fails.

    # Check if all models met both criteria by iterating through the calculated metrics
    if all_models_passed_criteria: # This flag would have been set if any model failed
        # If the loop above completed and all_models_passed_criteria is still true,
        # or if we explicitly check each stored metric now:
        pass # Continue to check if all actually passed. Redundant if flag is set correctly above.

    # Re-checking the condition explicitly for clarity and correctness based on stored metrics
    passed_mula = (models_metrics["MulaRec"]["achievement_rate"] >= threshold_achievement_rate and
                   models_metrics["MulaRec"]["head_ratio"] >= threshold_head_ratio)
    passed_cb = (models_metrics["CodeBERT"]["achievement_rate"] >= threshold_achievement_rate and
                 models_metrics["CodeBERT"]["head_ratio"] >= threshold_head_ratio)
    passed_ct5 = (models_metrics["CodeT5"]["achievement_rate"] >= threshold_achievement_rate and
                  models_metrics["CodeT5"]["head_ratio"] >= threshold_head_ratio)

    if passed_mula and passed_cb and passed_ct5:
        return models_metrics["MulaRec"]["sequence"]  # Return MulaRec's prediction
    else:
        return ['none.none']  # Reject if not all models passed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to save the output file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="Path to the API database JSON file")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    args = parser.parse_args()

    with open(args.file1, 'r') as f:
        codet5_taiL_prediction = [int(line.strip()) for line in f if line.strip()] # This variable is not used later
    with open(args.file2, 'r') as file2:
        codebert_model_predictions = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r') as file3:
        codet5_model_predictions = [parse_string_into_apis(pred) for pred in file3.readlines()]
    with open(args.file4, 'r') as file4:
        mularec_model_predictions = [parse_string_into_apis(pred) for pred in file4.readlines()]

    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    with open(args.api_database, 'r', encoding='utf-8') as file:
        api_database_json = json.load(file)
    api_dict_global = {
        entry["api_method"].lower().replace(' ', ''): entry 
        for entry in api_database_json
    }
    print(f"Loaded {len(api_dict_global)} entries into API database.")

    codebert_preds_mapped = []
    codet5_preds_mapped = []
    mularec_preds_mapped = []
    for index_val in original_index:
        codebert_preds_mapped.append(codebert_model_predictions[int(index_val)-1])
        codet5_preds_mapped.append(codet5_model_predictions[int(index_val)-1])
        mularec_preds_mapped.append(mularec_model_predictions[int(index_val)-1])
    
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
        else: # 3モデルの予測が全て異なる場合
            recommended_seq = get_recommendation_by_new_rules(cb_pred, ct5_pred, mula_pred, api_dict_global)
            final_apiseq_list.append(recommended_seq)

    accu_count = 0
    skip_count = 0
    num_testdata = len(correct_refs)

    for i in range(num_testdata):
        final_pred = final_apiseq_list[i]
        correct_api_seq = correct_refs[i]
        
        if 'none.none' in final_pred:
            skip_count += 1
        else:
            if final_pred == correct_api_seq:
                accu_count += 1
    
    if (num_testdata - skip_count) > 0:
        apiseq_final_accuracy = round((accu_count / (num_testdata - skip_count)) * 100, 2)
    else:
        apiseq_final_accuracy = 0.0 
    
    if num_testdata > 0:
        apiseq_final_rejection_rate = round((skip_count / num_testdata) * 100, 2)
    else:
        apiseq_final_rejection_rate = 0.0

    print(f"Final API Sequence Accuracy: {apiseq_final_accuracy}%")
    print(f"Final API Sequence Rejection Rate: {apiseq_final_rejection_rate}%")
    print(f"Total Processed: {num_testdata}, Correctly Recommended: {accu_count}, Rejected: {skip_count}")

if __name__ == '__main__':
    main()