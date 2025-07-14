"""APIメソッドシーケンス版モデルプロファイルを使用し、
信頼度スコアに基づいて最も信頼できるAPIメソッドシーケンスを推薦するスクリプト
信頼スコアは、正解実績率のみを使用して計算される。"""
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
# グラフの日本語表示
import japanize_matplotlib

def parse_string_into_apis(str_):
    apis = []
    # タブや前後の空白を処理し、ドットで分割
    eles = str_.split('\t')[0].strip().split('.')
    if not eles or not eles[0]:
        return []

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
            # スペースがない、または他の形式の場合
            module_ = eles[i].strip()
            api = first_lib.strip() + '.' + module_.strip()
            api = api.lower().replace(' ', '')
            apis.append(api)
            first_lib = module_

    # 最後の要素を追加
    if first_lib and eles[-1]:
        api = first_lib.strip() + '.' + eles[-1].strip()
        api = api.lower().replace(' ', '')
        apis.append(api)
    return apis


def get_plausible_apiseq(codebert_preds, codet5_preds, mularec_preds, api_dict, th=0.0):
    """
    各モデルのAPIメソッドシーケンスの「信頼度スコア」を計算し、
    最も高いスコアを持つモデルのシーケンスを返す。
    """
    def calculate_reliability_score(api_sequence, model_name):
        """信頼度スコア = 正解実績率 * ヘッド変数"""

        # ▼▼▼【修正点 1】APIシーケンスのリストをタプルに変換してキーとして使用 ▼▼▼
        seq_key = tuple(api_sequence)
        
        try:
            # シーケンス全体の統計データを取得
            seq_data = api_dict[seq_key]
            
            # モデルごとの正解率を計算
            appearance_count = seq_data.get(f"{model_name}_appearance_count", 0)
            correct_count = seq_data.get(f"{model_name}_correct_count", 0)

            if appearance_count == 0:
                ratio = 0.0
            else:
                ratio = correct_count / appearance_count
            
            # ヘッド変数 (シーケンス全体がヘッドなら1, テールなら0)
            head_variable = 1 if seq_data.get("head_or_tail", 1) == 0 else 0
            
            # 信頼度スコア
            score = ratio * head_variable
            return score

        except KeyError:
            # データベースにシーケンス情報がない場合はスコア0
            return 0.0
        # ▲▲▲【修正点 1】ここまで ▲▲▲

    # 各モデルの信頼度スコアを計算
    codebert_score = calculate_reliability_score(codebert_preds, "CodeBERT")
    codet5_score = calculate_reliability_score(codet5_preds, "CodeT5")
    mularec_score = calculate_reliability_score(mularec_preds, "MulaRec")

    # スコアの比較
    scores = {
        "CodeBERT": codebert_score,
        "CodeT5": codet5_score,
        "MulaRec": mularec_score,
    }
    highest_model = max(scores, key=scores.get)
    highest_score = scores[highest_model]

    # スコアがしきい値以下の場合は棄却
    if highest_score <= th:
        return ['none.none']

    # 最も高いスコアを持つモデルの予測を返す
    if highest_model == "CodeBERT":
        return codebert_preds
    elif highest_model == "CodeT5":
        return codet5_preds
    else: # MulaRec
        return mularec_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to save the output file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--api_database', type=str, default='./data/sequence_correctCount_and_appearCount30.json', help="whole")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    parser.add_argument('--output_file', type=str, default='./data/apiseq_recommendation_reliablescore_newprofile30.png', help="Path to save the output file")
    args = parser.parse_args()

    with open(args.file1, 'r') as f:
        codet5_taiL_prediction = [int(line.strip()) for line in f if line.strip()]
    with open(args.file2, 'r') as file2:
        codebert_predictions = [parse_string_into_apis(pred) for pred in file2.readlines()]
    with open(args.file3, 'r') as file3:
        codet5_predictions = [parse_string_into_apis(pred) for pred in file3.readlines()]
    with open(args.file4, 'r') as file4:
        mularec_predictions = [parse_string_into_apis(pred) for pred in file4.readlines()]

    tail_indices = [i for i, value in enumerate(codet5_taiL_prediction) if value == 1]
    print('Tail indices count: ', len(tail_indices))     

    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()
    original_index = df['api_index'].tolist()

    # APIデータベースを読み込む
    with open(args.api_database, 'r', encoding='utf-8') as file:
        api_database_json = json.load(file)

    # ▼▼▼【修正点 2】APIシーケンスのリストをタプルに変換して辞書のキーにする ▼▼▼
    api_dict = {
        tuple(entry["api_sequence"]): entry 
        for entry in api_database_json if entry.get("api_sequence")
    }
    # ▲▲▲【修正点 2】ここまで ▲▲▲

    codebert_preds = []
    codet5_preds = []
    mularec_preds = []
    for index in original_index:
        codebert_preds.append(codebert_predictions[int(index)-1])
        codet5_preds.append(codet5_predictions[int(index)-1])
        mularec_preds.append(mularec_predictions[int(index)-1])
    
    thresholds = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
    apiseq_accuracy_list = []
    apiseq_rejection_rate_list = []

    for th in thresholds:
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
                plausible_apimethod = get_plausible_apiseq(codebert_pred, codet5_pred, mularec_pred, api_dict, th)
                final_apiseq.append(plausible_apimethod)

        accu_count = 0
        skip_count = 0
        num_testdata = len(correct_refs)
        for i in range(num_testdata):
            final_pred = final_apiseq[i]
            api_seq = correct_refs[i]
            if 'none.none' in final_pred:
                skip_count += 1
            else:
                if final_pred == api_seq:
                    accu_count += 1
        
        effective_recommendations = num_testdata - skip_count
        if effective_recommendations > 0:
            apiseq_final_accuracy = round((accu_count / effective_recommendations)*100, 2)
        else:
            apiseq_final_accuracy = 0.0

        if num_testdata > 0:
            apiseq_final_rejection_rate = round(((skip_count / num_testdata))*100, 2)
        else:
            apiseq_final_rejection_rate = 0.0
            
        print(f"Threshold: {th}")
        print(f"  Accuracy: {apiseq_final_accuracy}%")
        print(f"  Rejection Rate: {apiseq_final_rejection_rate}%")
        print(f"  (Correct: {accu_count}, Rejected: {skip_count}, Total: {num_testdata})")
        apiseq_accuracy_list.append(apiseq_final_accuracy)
        apiseq_rejection_rate_list.append(apiseq_final_rejection_rate)

    # グラフの作成
    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, apiseq_accuracy_list, label="正解率 (APIシーケンス推薦)", color="green", marker="s", linestyle="--")
    plt.plot(thresholds, apiseq_rejection_rate_list, label="棄却率 (APIシーケンス推薦)", color="red", marker="s", linestyle="--")

    plt.xlabel("出力棄却判定のしきい値（信頼度スコア）", fontsize=14)
    plt.ylabel("パーセンテージ (%)", fontsize=14)
    plt.xticks(np.arange(0.9, 1.01, 0.01), fontsize=12)
    plt.yticks(np.arange(0, max(max(apiseq_accuracy_list, default=0), max(apiseq_rejection_rate_list, default=0)) + 10, 10), fontsize=12)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    plt.savefig(args.output_file, dpi=300)
    plt.show()
    
if __name__ == '__main__':
    main()