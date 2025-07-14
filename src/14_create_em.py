import pandas as pd
import argparse
import json
import os

def em_index_list(model_preds, correct_refs):
    index_list = []
    for index, (pred_seq, ref_seq) in enumerate(zip(model_preds, correct_refs)):
        if pred_seq == ref_seq:
            index_list.append(index+1)
    return index_list

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
    return apis

def write_matching_indices(file_path, indices):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index in indices:
            file.write(f"{index}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_predictions.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_predictions.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_predictions.txt', help="Path to save the output file")
    parser.add_argument('--test_file', type=str, default='./data/test_3_lines.csv', help="Path to save the output file")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Path to save the text file which including tail data")
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
    
    # 各モデルの正解数をカウントする
    codebert_index_list = em_index_list(codebert_preds_file, correct_refs)
    codet5_index_list = em_index_list(codet5_preds_file, correct_refs)
    mularec_index_list = em_index_list(mularec_preds_file, correct_refs)

    # 最終結果の出力
    codebert_output_file = os.path.join(args.output_dir, 'CodeBERT_exactmatch_index.txt')
    write_matching_indices(codebert_output_file, codebert_index_list)
    codet5_output_file = os.path.join(args.output_dir, 'CodeT5_exactmatch_index.txt')
    write_matching_indices(codet5_output_file, codet5_index_list)
    mularec_output_file = os.path.join(args.output_dir, 'MulaRec_exactmatch_index.txt')
    write_matching_indices(mularec_output_file, mularec_index_list)

    print(f"CodeBERT ExactMatch count: {len(codebert_index_list)}")
    print(f"CodeT5 ExactMatch count: {len(codet5_index_list)}")
    print(f"MulaRec ExactMatch count: {len(mularec_index_list)}")

    

if __name__ == "__main__":
    main()