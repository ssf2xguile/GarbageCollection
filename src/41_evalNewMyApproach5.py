"""上位5件APIメソッドを基にした新しい推薦ルールを適用し、APIシーケンスの正解率と棄却率を計算するスクリプト"""
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
    accuracy_product = 1.0
    seq_len = len(api_sequence)

    if seq_len == 0:
        return 0.0

    for api in api_sequence:
        try:
            api_data = api_dict_local[api] # api_dict_localのキーは正規化済みと仮定
            correct_count = api_data[f"{model_name}_correct_count"]
            # JSON構造と過去の議論に基づき "appearance_count" を使用
            appearance_count = api_data["appearance_count"] 
            
            if appearance_count == 0:
                accuracy_product = 0.0  # 出現がない場合は寄与0
                break # 積が0になるためループを抜ける
            else:
                ratio = correct_count / appearance_count
                accuracy_product *= ratio
                if accuracy_product == 0.0: # 積が0になったら早期終了
                    break
        except KeyError:
            accuracy_product = 0.0  # API情報がない場合も寄与0
            break # ループを抜ける
            
    if accuracy_product == 0.0:
        return 0.0
        
    return math.pow(accuracy_product, 1.0 / seq_len)

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

# モデルごとの上位5件APIメソッドリスト
TOP_5_APIS = {
    "MulaRec": ["string.length", "string.substring", "stringbuffer.append", "integer.parseint", "string.equals"],
    "CodeBERT": ["list.add", "list.get", "string.length", "integer.parseint", "list.size"],
    "CodeT5": ["iterator.next", "string.cast", "stringbuffer.append", "stringbuffer.tostring", "entry.cast"]
}

def get_recommendation_by_new_rules(cb_pred, ct5_pred, mula_pred, api_dict_local): # api_dict_local は現状この関数では未使用だが、将来的な拡張のために残すことも可能
    """新しいルール（上位5件APIメソッド）に基づいて推薦APIシーケンスを決定する"""
    
    # 評価するモデルと予測のリスト（優先順位順）
    models_to_evaluate = [
        ("MulaRec", mula_pred),
        ("CodeBERT", cb_pred),
        ("CodeT5", ct5_pred)
    ]

    for model_name, pred_sequence in models_to_evaluate:
        model_top_apis = TOP_5_APIS.get(model_name, [])
        if not model_top_apis: #念のため、モデル名がTOP_5_APISにない場合
            continue

        # 予測シーケンスに上位5件APIメソッドのいずれかが含まれているかチェック
        # setを使うと効率的
        pred_set = set(pred_sequence)
        top_api_set = set(model_top_apis)
        
        if not pred_set.isdisjoint(top_api_set): # 共通要素が1つでもあればTrue
            return pred_sequence # 条件を満たした最初のモデルの予測を返す

    return ['none.none'] # どのモデルも条件を満たさなかった場合は棄却

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file") # 未使用のためコメントアウト
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the CodeBERT predictions file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to the CodeT5 predictions file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the MulaRec predictions file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="Path to the API database JSON file")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test CSV file")
    args = parser.parse_args()

    # モデルの予測を読み込み
    with open(args.file2, 'r') as file2:
        codebert_model_predictions = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r') as file3:
        codet5_model_predictions = [parse_string_into_apis(pred) for pred in file3.readlines()]
    with open(args.file4, 'r') as file4:
        mularec_model_predictions = [parse_string_into_apis(pred) for pred in file4.readlines()]

    # 正解データを読み込み
    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist() # test_file のインデックス

    # APIデータベースを読み込み、キーを正規化
    with open(args.api_database, 'r', encoding='utf-8') as file:
        api_database_json = json.load(file)
    api_dict_global = {
        entry["api_method"].lower().replace(' ', ''): entry 
        for entry in api_database_json
    }
    print(f"Loaded {len(api_dict_global)} entries into API database.")

    # original_index に基づいて各モデルの予測をマッピング
    codebert_preds_mapped = []
    codet5_preds_mapped = []
    mularec_preds_mapped = []
    # original_index は1から始まると仮定
    for index_val in original_index:
        actual_idx = int(index_val) - 1 # 0-based index for list access
        codebert_preds_mapped.append(codebert_model_predictions[actual_idx])
        codet5_preds_mapped.append(codet5_model_predictions[actual_idx])
        mularec_preds_mapped.append(mularec_model_predictions[actual_idx])
    
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
            # api_dict_global は get_recommendation_by_new_rules で現在未使用だが、引数として渡す形は維持
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
    
    # 正解率と棄却率の計算
    effective_recommendations = num_testdata - skip_count
    if effective_recommendations > 0:
        apiseq_final_accuracy = round((accu_count / effective_recommendations) * 100, 2)
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
