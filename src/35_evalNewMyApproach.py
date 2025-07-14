import pandas as pd
import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

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


def get_plausible_apiseq(codebert_preds, codet5_preds, mularec_preds, api_dict, th=0.0):
    """
    各モデルのAPIメソッドシーケンスの正解率を計算し、
    最も高い正解率を持つモデルのシーケンスを返す。
    """
    def calculate_sequence_accuracy(api_sequence, model_name):
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

    # 各モデルのAPIシーケンスの正解率を計算
    codebert_accuracy = calculate_sequence_accuracy(codebert_preds, "CodeBERT")
    codet5_accuracy = calculate_sequence_accuracy(codet5_preds, "CodeT5")
    mularec_accuracy = calculate_sequence_accuracy(mularec_preds, "MulaRec")

    # 正解率の比較
    accuracies = {
        "CodeBERT": codebert_accuracy,
        "CodeT5": codet5_accuracy,
        "MulaRec": mularec_accuracy,
    }
    highest_model = max(accuracies, key=accuracies.get)  # 最も高い正解率を持つモデル名
    highest_accuracy = accuracies[highest_model]

    # 正解率が0の場合は ['none.none'] を返す    閾値はデフォルト0.0であるが、自由に変更できる
    if highest_accuracy <= th:
        return ['none.none']

    # 最も高い正解率を持つモデルの予測を返す
    if highest_model == "CodeBERT":
        return codebert_preds
    elif highest_model == "CodeT5":
        return codet5_preds
    else:
        return mularec_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/codet5_tail_prediction_90.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/CodeT5_predictions.txt', help="Path to save the output file")
    parser.add_argument('--file4', type=str, default='./data/MulaRec_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--api_database', type=str, default='./data/first_correctCount_and_appearCount.json', help="whole")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to the test.csv file")
    parser.add_argument('--output_file', type=str, default='./data/apiseq_recommendation_reliablescore.png', help="Path to save the output file")
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
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    apiseq_accuracy_list = []
    apiseq_rejection_rate_list = []
    for index in original_index:
        codebert_preds.append(codebert_predictions[int(index)-1])
        codet5_preds.append(codet5_predictions[int(index)-1])
        mularec_preds.append(mularec_predictions[int(index)-1])
    
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
        apiseq_accuracy_list.append(apiseq_final_accuracy)
        apiseq_rejection_rate_list.append(apiseq_final_rejection_rate)

    # グラフの作成
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, apiseq_accuracy_list, label="正解率 (APIメソッドシーケンス推薦)", color="green", marker="s", linestyle="--")
    plt.plot(thresholds, apiseq_rejection_rate_list, label="棄却率 (APIメソッドシーケンス推薦)", color="red", marker="s", linestyle="--")

    # 軸ラベルとタイトル
    plt.xlabel("出力棄却判定のしきい値", fontsize=14)
    plt.ylabel("パーセンテージ (%)", fontsize=14)

    # 目盛りのフォントサイズ
    plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
    plt.yticks(fontsize=12)

    # 凡例のフォントサイズ
    plt.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0, 0.25))
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