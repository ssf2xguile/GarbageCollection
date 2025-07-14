import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='./data/api_model_correct_count.json', help="Path to the JSON file containing API correct counts")
    parser.add_argument('--output_dir', type=str, default='./data/', help="Directory to save the Venn diagram images")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for the saved images (higher means better quality)")
    args = parser.parse_args()

    # JSONファイルを読み込む
    with open(args.json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 各モデルの正解APIをセットに格納（ヘッドとテールに分ける）
    head_codebert_correct = set()
    head_codet5_correct = set()
    head_mularec_correct = set()

    tail_codebert_correct = set()
    tail_codet5_correct = set()
    tail_mularec_correct = set()

    head_count = 0
    tail_count = 0

    for entry in data:
        if entry["head_or_tail"] == 0:  # ヘッド
            head_count += 1
            if entry["CodeBERT_correct_count"] > 0:
                head_codebert_correct.add(entry["api_method"])
            if entry["CodeT5_correct_count"] > 0:
                head_codet5_correct.add(entry["api_method"])
            if entry["MulaRec_correct_count"] > 0:
                head_mularec_correct.add(entry["api_method"])
        elif entry["head_or_tail"] == 1:  # テール
            tail_count += 1
            if entry["CodeBERT_correct_count"] > 0:
                tail_codebert_correct.add(entry["api_method"])
            if entry["CodeT5_correct_count"] > 0:
                tail_codet5_correct.add(entry["api_method"])
            if entry["MulaRec_correct_count"] > 0:
                tail_mularec_correct.add(entry["api_method"])

    # ヘッドとテールの種類数を標準出力
    print(f"ヘッドのAPIメソッドの種類数: {head_count}")
    print(f"テールのAPIメソッドの種類数: {tail_count}")
    print(f"1つも正解しなかったテールのAPIメソッドの種類数: {tail_count - len(tail_codebert_correct.union(tail_codet5_correct, tail_mularec_correct))}")

    # ヘッドのベン図の描画と保存
    plt.figure(figsize=(8, 8))
    venn_diagram_head = venn3([head_codebert_correct, head_codet5_correct, head_mularec_correct],
                              set_labels=('CodeBERT', 'CodeT5', 'MulaRec'))
    plt.title("Venn Diagram of Head API Methods across Models")
    head_output_path = os.path.join(args.output_dir, 'head_api_venn_diagram.png')
    plt.savefig(head_output_path, dpi=args.dpi)
    print(f"Head Venn diagram saved to {head_output_path}")
    plt.show()

    # テールのベン図の描画と保存
    plt.figure(figsize=(8, 8))
    venn_diagram_tail = venn3([tail_codebert_correct, tail_codet5_correct, tail_mularec_correct],
                              set_labels=('CodeBERT', 'CodeT5', 'MulaRec'))
    plt.title("Venn Diagram of Tail API Methods across Models")
    tail_output_path = os.path.join(args.output_dir, 'tail_api_venn_diagram.png')
    plt.savefig(tail_output_path, dpi=args.dpi)
    print(f"Tail Venn diagram saved to {tail_output_path}")
    plt.show()

if __name__ == "__main__":
    main()
