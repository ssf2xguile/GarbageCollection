import json
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import json
import argparse
import os
from time import time
import json
from collections import Counter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import pandas as pd
import json

def gini_coef(wealths):
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float64(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return (A / (A+B))

def parse_string_into_apis(str_):
    apis = []
    eles = str_.split('.')

    first_lib = eles[0]

    for i in range(1, len(eles)-1):
        try:
            module_, library_ = eles[i].strip().rsplit(' ')
        except:
            module_, library_ = eles[i].strip().split(' ', 1)
        apis.append(first_lib.strip()+'.'+module_.strip())
        first_lib = library_

    apis.append(first_lib.strip() +'.'+ eles[-1].strip())
    return apis

def main():
    data_dir = './data/'

    ###API class Data   例)['HashSet . <init> ... String . charAt', 'Error . <init>', 'Util . contains Message . src', ...]
    splits = ['train', 'test','validate']
    splits = ['test']
    all_data = []
    for split in splits:
        file_name = data_dir + split + '_3_lines.csv'
        df = pd.read_csv(file_name)
        all_data.append(df)
    all_data = pd.concat(all_data)
    all_data = list(all_data['target_api'])
    print(len(all_data))
    all_api_data = []
    
    # 各行の正解APIをまとめてリストに格納   '[hashset.<init>', ..., 'string.charat', 'error.<init>', 'util.contains', 'message.src', ...]
    for d in all_data:
        api_seqs = parse_string_into_apis(d)
        api_seqs = [a.lower() for a in api_seqs]
        all_api_data.extend(api_seqs)
    all_data = all_api_data

    # 頻出度の高いAPIメソッドとその回数をリストに格納    例) vocab_tokens=['grouplayout.createparallelgroup', 'grouplayout.createsequentialgroup', ..., 'jlabel.<init>'], vacab_samples=[5441, 3972, ..., 1887]
    api_rec_ids = list(range(len(all_data)))
    api_rec_labels = all_data
    api_rec_categories = ['api_sequence_rec']*len(all_data)
    vocab = Counter(api_rec_labels)
    vocab_tokens = [i[0] for i in vocab.most_common(len(vocab))]
    vocab_samples = [i[1] for i in vocab.most_common(len(vocab))]
    total_sample = sum(vocab_samples)
    print(vocab_samples[:10])

    label2label_num_dict = {}
    for i in range(len(vocab_tokens)):
        label2label_num_dict[vocab_tokens[i]] = i
    api_rec_labels = [ label2label_num_dict[l] for l in api_rec_labels]
    api_rec_df = {'id': api_rec_ids, 'Sorted ID': api_rec_labels, 'task': api_rec_categories   }
    df = pd.DataFrame(api_rec_df)

    
    # # Draw Plot
    fig_scale = 0.4
    plt.figure(figsize=(int(20*fig_scale),int(10*fig_scale)), dpi= int(80/fig_scale))
    sns.distplot(df.loc[df['task'] == 'api_sequence_rec', "Sorted ID"], color=sns.color_palette("husl", 8)[0], label="api_sequence_rec",  hist_kws={'alpha':.7}, kde_kws={'linewidth':2})
    plt.ylim(0, 0.00045)
    plt.xlim(0, )
    plt.xlabel('Sorted Label ID', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.tight_layout()
    plt.show()
    

    vocab_taking_up_ratios = [float(l / total_sample) for l in vocab_samples]
    print('Gini coeffcient:', gini_coef(vocab_taking_up_ratios))




if __name__ == "__main__":
    main()

# --task=api --data_dir='../all_data/api_seq_data/mularec_data/'
# --task=revision --data_dir='../all_data/code_review_data/codet5_data/codet5_format_data/refine/small/'
# --task=vulnerability --data_dir='../all_data/vulnerability_data/dataset/'