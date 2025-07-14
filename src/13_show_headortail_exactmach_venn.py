# 3つのLLMによるExactmatchしたテストデータのうち、テールデータを含むテストデータについてのベン図を示す
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', type=str, default='./data/CodeBERT_exactmatch_index_investigate.txt', help="Path to the prediction.txt file")
    parser.add_argument('--file2', type=str, default='./data/CodeT5_exactmatch_index_investigate.txt', help="Path to the test.json file")
    parser.add_argument('--file3', type=str, default='./data/MulaRec_exactmatch_index_investigate.txt', help="Path to save the output file")
    parser.add_argument('--output', type=str, default='./data/Exactmatch_venn_for_DSW.png', help="Path to save the Venn diagram image")
    parser.add_argument('--fontsize', type=int, default=20, help="Font size for the numbers in the Venn diagram")
    parser.add_argument('--dpi', type=int, default=200, help="DPI for the saved image (higher means better quality)")
    args = parser.parse_args()

    # Read the content from the files and convert to sets of integers
    with open(args.file1, 'r') as file1, open(args.file2, 'r') as file2, open(args.file3, 'r') as file3:
        set1 = set(map(int, file1.read().strip().split()))
        set2 = set(map(int, file2.read().strip().split()))
        set3 = set(map(int, file3.read().strip().split()))

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