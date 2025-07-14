"""
このプログラムは、予測テキストファイル (prediction.txt) と指定されたテストファイル（.json、.fixed、または .csv）の各行を比較し、完全一致する行番号を新しいテキストファイルに書き込むもの。テストファイルの形式に応じて適切な読み込み処理を行い、一致する行のインデックスを探す。また、コマンドラインで一致した行数を表示する。
【入出力】
入力
・予測ファイル (prediction.txt): 各行に予測結果が記述されたテキストファイル。
・テストファイル (test_file): 比較対象となるファイル。.json、.fixed、または .csv 形式。
出力
・出力ファイル (matching_indices.txt): prediction.txt の行とテストファイルの行が完全一致した行番号を格納したファイル。
【実行方法】
例) python investigate_correct_index.py --prediction_file prediction.txt --test_file test.csv --output_file matching_indices.txt
"""

import json
import argparse
import os
import pandas as pd
import re

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 2個以上の連続する空白を1個の空白に置き換える
    lines = [re.sub(r'\s+', ' ', line) for line in lines]
    return [line.strip() for line in lines]

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    lines = [' '.join(item['docstring_tokens']) for item in data]
    # 2個以上の連続する空白を1個の空白に置き換える
    lines = [re.sub(r'\s+', ' ', line) for line in lines]
    return lines

def read_fixed_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def read_csv_file(file_path):
    df = pd.read_csv(file_path)
    return df['target_api'].str.lower()  # Exclude missing values

def check_matching_lines(txt_lines, test_lines):    #txt_lines:予測結果のAPIメソッドのリスト["Utility . getValidChoice ... Scanner . nextLine", "Error . <init> ... ExceptionInInitializer . error", ...], test_lines:テストデータのAPIメソッドのリスト["HashSet . <init> ... String . charAt", ]
    matching_indices = []
    all_txt_apimethod_list = [] #予測結果のAPIメソッドのリストを格納するリスト [["utility . getvalidchoice", ... "scanner . nextline"], ["error . <init>", ... "exceptionininitializer . error"], ...]
    all_test_apimethod_list = [] #テストデータのAPIメソッドのリストを格納するリスト [["hashset . <init>", ... "string . charat"], ...]
    for i, (txt_line, test_line) in enumerate(zip(txt_lines, test_lines)):
        txt_apimethod_list = parse_string_into_apis(txt_line)
        test_apimethod_list = parse_string_into_apis(test_line)
        txt_apimethod_list = [a.lower() for a in txt_apimethod_list]
        test_apimethod_list = [a.lower() for a in test_apimethod_list]
        all_txt_apimethod_list.append(txt_apimethod_list)
        all_test_apimethod_list.append(test_apimethod_list)

    for i, (txt_apimethod, test_apimethod) in enumerate(zip(all_txt_apimethod_list, all_test_apimethod_list)):
        if (len(txt_apimethod) == len(test_apimethod)):  # 両方のAPIメソッドの数が等しい場合
            if txt_apimethod == test_apimethod: # 全てのAPIメソッドが一致している場合
                matching_indices.append(i+1)

    return matching_indices

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
            apis.append(first_lib.strip()+'.'+module_.strip())
            first_lib = library_
        except ValueError:
            try:
                module_, library_ = eles[i].strip().split(' ', 1)
                apis.append(first_lib.strip()+'.'+module_.strip())
                first_lib = library_
            except ValueError:  # splitで失敗した場合の処理 例えばPoint . Math . pow Math . sqrtのようにドットが繋がっている場合
                module_ = eles[i].strip()
                library_ = ''
                apis.append(first_lib.strip()+'.'+module_.strip())
                first_lib = module_

    apis.append(first_lib.strip() +'.'+ eles[-1].strip())
    return apis

def write_matching_indices(file_path, indices):
    with open(file_path, 'w', encoding='utf-8') as file:
        for index in indices:
            file.write(f"{index}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default='./data/CodeT5_predictions.txt',help="Path to the prediction.txt file")
    parser.add_argument('--test_file', type=str, default='./data/test.buggy-fixed.fixed', help="Path to the test.json file")
    parser.add_argument('--output_file', type=str, default='./data/CodeT5_exactmatch_index.txt', help="Path to save the output file")
    args = parser.parse_args()

    txt_lines = read_txt_file(args.prediction_file)

    # Determine the file extension and read accordingly
    _, ext = os.path.splitext(args.test_file)
    if ext == '.json':
        test_lines = read_json_file(args.test_file)
    elif ext == '.fixed':
        test_lines = read_fixed_file(args.test_file)
    elif ext == '.csv':
        test_lines = read_csv_file(args.test_file)
    else:
        raise ValueError("Unsupported file extension. Only .json and .fixed files are supported.")

    matching_indices = check_matching_lines(txt_lines, test_lines)

    write_matching_indices(args.output_file, matching_indices)

    # Output the number of matches
    total_lines = len(txt_lines)
    matched_lines = len(matching_indices)
    print(f"{matched_lines} out of {total_lines} lines matched.")
    print(f"Matching indices written to {args.output_file}")

if __name__ == "__main__":
    main()
