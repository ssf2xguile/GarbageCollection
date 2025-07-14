"""信頼スコアと算術平均
信頼度スコアは、各APIメソッドの正解実績率とヘッド変数を掛け合わせた幾何平均で計算されます。
信頼度スコアがしきい値以下の場合は、'none.none'を返します。"""
import pandas as pd
import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
# グラフの日本語表示が必要な場合はコメントを解除してください
import japanize_matplotlib

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


def get_plausible_apiseq(codebert_preds, codet5_preds, mularec_preds, api_dict, th=0.0):
    """
    各モデルのAPIメソッドシーケンスの「信頼度スコア」を計算し、
    最も高いスコアを持つモデルのシーケンスを返す。
    """
    def calculate_reliability_score(api_sequence, model_name):
        """信頼度スコア = 平均(正解実績率 * ヘッド変数)"""
        weighted_scores = []
        seq_len = len(api_sequence)
        
        for api in api_sequence:
            try:
                api_data = api_dict[api]
                correct_count = api_data[f"{model_name}_correct_count"]
                # 'first_appearance_count'を使用。ファイルに合わせて'first_appearance_count'に変更が必要な場合がある
                first_appearance_count = api_data.get("first_appearance_count", 0)

                # 正解実績率の計算
                if first_appearance_count == 0:
                    ratio = 0.0
                else:
                    ratio = correct_count / first_appearance_count
                
                # ヘッド変数の決定 (ヘッドAPIなら1, テールAPIなら0)
                # head_or_tail == 0 をヘッドAPIと仮定
                head_variable = 1 if api_data.get("head_or_tail", 1) == 0 else 0

                # 重み付けされたスコア（正解実績率 * ヘッド変数）を追加
                weighted_scores.append(ratio * head_variable)

            except KeyError:
                weighted_scores.append(0.0) # API情報がない場合はスコア0
        
        # 算術平均を計算
        return sum(weighted_scores) / seq_len

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
    highest_model = max(scores, key=scores.get)  # 最も高いスコアを持つモデル名
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
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="whole")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    parser.add_argument('--output_file', type=str, default='./data/apiseq_recommendation_reliablescore_rule12.png', help="Path to save the output file")
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
            api_database_json = json.load(file)
    # APIメソッド名をキー、詳細を値とする辞書に変換（キーを正規化）
    api_dict = {
        entry["api_method"].lower().replace(' ', ''): entry 
        for entry in api_database_json
    }

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
        num_testdata = len(correct_refs) # テストデータ総数をdfから取得する方が堅牢
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
            apiseq_final_accuracy = round((accu_count / effective_recommendations)*100,2)
        else:
            apiseq_final_accuracy = 0.0

        if num_testdata > 0:
            apiseq_final_rejection_rate = round(((skip_count / num_testdata))*100,2)
        else:
            apiseq_final_rejection_rate = 0.0
            
        print(f"Threshold: {th}")
        print(f"  Accuracy: {apiseq_final_accuracy}%")
        print(f"  Rejection Rate: {apiseq_final_rejection_rate}%")
        print(f"  (Correct: {accu_count}, Rejected: {skip_count}, Total: {num_testdata})")
        apiseq_accuracy_list.append(apiseq_final_accuracy)
        apiseq_rejection_rate_list.append(apiseq_final_rejection_rate)

    # グラフの作成
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, apiseq_accuracy_list, label="正解率 (APIメソッドシーケンス推薦)", color="green", marker="s", linestyle="--")
    plt.plot(thresholds, apiseq_rejection_rate_list, label="棄却率 (APIメソッドシーケンス推薦)", color="red", marker="s", linestyle="--")

    # 軸ラベルとタイトル
    plt.xlabel("出力棄却判定のしきい値（信頼度スコア）", fontsize=14)
    plt.ylabel("パーセンテージ (%)", fontsize=14)

    # 目盛りのフォントサイズ
    plt.xticks(np.arange(0.9, 1.01, 0.01), fontsize=12)
    plt.yticks(fontsize=12)

    # 凡例のフォントサイズ
    plt.legend(fontsize=12, loc="best")
    plt.grid(True)

    # 余白調整
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

    # 画像の保存
    #plt.savefig(args.output_file, dpi=300)

    # グラフの表示
    plt.show()
    
if __name__ == '__main__':
    main()
