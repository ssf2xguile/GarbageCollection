# やってることは17_test_exactmatch_whole_byTargetInvestigate.pyと同じ。違うのはテストデータで、各モデルが正解したインデックスを取得すること。
import pandas as pd
import argparse
import json
import os
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

def compare_apimethod_seq(model_preds, correct_refs):
    correct_index = []
    for index, (pred_seq, ref_seq) in enumerate(zip(model_preds, correct_refs)):
        if pred_seq == ref_seq:
            correct_index.append(index)
    return correct_index

def parse_string_into_apis(str_):   # 今回は小文字化と空白削除の処理を追加している。
    apis = []
    eles = str_.split('\t')[0].strip().split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            api = first_lib.strip()+'.'+module_.strip()
            api = api.lower().replace(' ','')
            apis.append(api)
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                api = first_lib.strip()+'.'+module_.strip()
                api = api.lower().replace(' ','')
                apis.append(api)
                first_lib = library_
            except ValueError:  # splitで失敗した場合の処理 例えばPoint . Math . pow Math . sqrtのようにドットが繋がっている場合
                module_ = eles[i].strip()
                library_ = ''
                api = first_lib.strip()+'.'+module_.strip()
                api = api.lower().replace(' ','')
                apis.append(api)
                first_lib = module_

    api = first_lib.strip()+'.'+eles[-1].strip()
    api = api.lower().replace(' ','')
    apis.append(api)
    return apis[0]

def write_matching_indices(file_path, indices):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index in indices:
            file.write(f"{index}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_predictions.txt', help="Path to save the output file")
    parser.add_argument('--test_file', type=str, default='./data/target_test.csv', help="Path to save the output file")
    parser.add_argument('--output', type=str, default='./data/Exactmatch_venn_apimethod_for_testdata.png', help="Path to save the Venn diagram image")
    parser.add_argument('--fontsize', type=int, default=20, help="Font size for the numbers in the Venn diagram")
    parser.add_argument('--dpi', type=int, default=200, help="DPI for the saved image (higher means better quality)")
    args = parser.parse_args()

    # 予測結果を読み込む
    with open(args.file1, 'r', encoding='utf-8') as file1:
        codebert_preds_file = [parse_string_into_apis(pred) for pred in file1.readlines()]
    with open(args.file2, 'r', encoding='utf-8') as file2:
        codet5_preds_file = [parse_string_into_apis(pred) for pred in file2.readlines()]    # [['base64.decodetoobject'], ['math.sqrt', 'point.calculatedistance', 'math.sqrt'], ['thread.start'],...]のようになる 
    with open(args.file3, 'r', encoding='utf-8') as file3:
        mularec_preds_file = [parse_string_into_apis(pred) for pred in file3.readlines()]   # [['base64.decodetoobject'], ['point.math', 'math.abs', 'point.distance'], ['thread.start'],...]のようになる

    # 正解データを加工する
    df = pd.read_csv(args.test_file)
    correct_refs = df['target_api'].apply(parse_string_into_apis).tolist()  # この処理によってリストのリストが得られる[['base64.decodetoobject'], ['math.sqrt', 'point.calculatedistance', 'math.sqrt'], ['thread.start'],...]
    original_index = df['api_index'].tolist()   # 元のテストデータの何番目のデータかを示すインデックス[..., 36149, 36154, ...]のようになる 配列はcorrect_refsと同じ長さ
    
    codebert_preds = []
    codet5_preds = []
    mularec_preds = []
    for index in original_index:
        codebert_preds.append(codebert_preds_file[int(index)-1])
        codet5_preds.append(codet5_preds_file[int(index)-1])
        mularec_preds.append(mularec_preds_file[int(index)-1])

    #print(mularec_preds[:5])
    # 各モデルの正解数をカウントする
    codebert_correct_index = compare_apimethod_seq(codebert_preds, correct_refs)
    codet5_correct_index = compare_apimethod_seq(codet5_preds, correct_refs)
    mularec_correct_index = compare_apimethod_seq(mularec_preds, correct_refs)

    set1 = set(map(int, codebert_correct_index))
    set2 = set(map(int, codet5_correct_index))
    set3 = set(map(int, mularec_correct_index))

    # Create a Venn diagram using the sets
    plt.figure(figsize=(8, 8))
    venn_diagram = venn3([set1, set2, set3], ('CodeBERT', 'CodeT5', 'MulaRec'))

    # Change font size of the numbers in the Venn diagram
    for subset_id in venn_diagram.subset_labels:
        if subset_id:  # Avoid NoneType errors
            subset_id.set_fontsize(args.fontsize)
    for label_id in venn_diagram.set_labels:
        label_id.set_fontsize(args.fontsize)
    # Save the figure with high resolution
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight', pad_inches=0)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()