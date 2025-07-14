import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

# データ
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

rejection_rates_singleapimethod = [27.0, 27.0, 27.0, 27.1, 27.2, 27.7, 28.8, 29.7, 30.5, 31.2, 32.8]
accuracy_rates_singleapimethod = [65.9, 65.9, 65.9, 66.0, 66.0, 66.4, 67.2, 68.1, 68.7, 69.2, 70.6]

rejection_rates_apimethodseq = [43.9, 43.9, 43.9, 43.9, 44.0, 44.1, 44.5, 44.9, 45.4, 45.8, 46.1]
accuracy_rates_apimethodseq = [52.9, 52.9, 52.9, 52.9, 52.9, 53.1, 53.4, 53.8, 54.2, 54.5, 57.8]

# グラフの作成
plt.figure(figsize=(8, 6))
plt.plot(thresholds, accuracy_rates_singleapimethod, label="正解率 (単一APIメソッド推薦)", color="blue", marker="o", linestyle="-")
plt.plot(thresholds, rejection_rates_singleapimethod, label="棄却率 (単一APIメソッド推薦)", color="orange", marker="o", linestyle="-")
plt.plot(thresholds, accuracy_rates_apimethodseq, label="正解率 (APIメソッドシーケンス推薦)", color="green", marker="s", linestyle="--")
plt.plot(thresholds, rejection_rates_apimethodseq, label="棄却率 (APIメソッドシーケンス推薦)", color="red", marker="s", linestyle="--")

# 軸ラベルとタイトル
plt.xlabel("出力棄却判定のしきい値", fontsize=14)
plt.ylabel("パーセンテージ (%)", fontsize=14)

# 目盛りのフォントサイズ
plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
plt.yticks(fontsize=12)

# 凡例のフォントサイズ
plt.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0, 0.1))
plt.grid(True)

# 余白調整
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

# 画像の保存
plt.savefig("method_recommendation_comparison.png", dpi=300)

# グラフの表示
plt.show()