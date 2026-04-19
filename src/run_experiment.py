import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from clip_utils import load_clip_model, get_similarity


def main():
    # 1. モデルのロード
    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model, preprocess = load_clip_model(MODEL_NAME, DEVICE)

    # 2. 類似度（確率分布）の計算
    print(f"Running experiment on image: {TARGET_IMAGE}")
    probs = get_similarity(model, preprocess, TARGET_IMAGE, LABEL_SET, DEVICE)

    # 3. 結果をDataFrameにまとめて整理
    df = pd.DataFrame({
        "label": LABEL_SET,
        "probability": probs
    }).sort_values(by="probability", ascending=False)

    # --- 結果の保存とグラフ化 ---

    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_name = os.path.basename(TARGET_IMAGE).split('.')[0]

    #  CSVとして保存
    csv_path = os.path.join(OUTPUT_DIR, f"results_{image_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved results to: {csv_path}")

    #  可視化（Seabornによる棒グラフ）
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    import matplotlib as mpl
    # Windowsの絵文字フォントを指定
    mpl.rcParams['font.family'] = ['Segoe UI Emoji', 'DejaVu Sans']

    # グラフの描画
    ax = sns.barplot(x="probability", y="label", data=df, palette="viridis")

    plt.title(f"CLIP Zero-shot Analysis: {image_name}", fontsize=14)
    plt.xlabel("Probability", fontsize=12)
    plt.ylabel("Prompts", fontsize=12)
    plt.xlim(0, 1.0)  # 確率なので0~1の範囲に固定

    # 数値を棒の横に表示
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)

    plt.tight_layout()

    # グラフの保存
    graph_path = os.path.join(OUTPUT_DIR, f"graph_{image_name}.png")
    plt.savefig(graph_path)
    print(f"Saved graph to: {graph_path}")

    # 画面に表示
    plt.show()


if __name__ == "__main__":
    main()