# CLIP Robustness Evaluation with "Weird" Labels

## 概要

本プロジェクトは、OpenAIの CLIP (Contrastive Language-Image Pre-training) を用い、入力プロンプト（ラベル）の表現を意図的に変化させた際のゼロショット分類の挙動を検証したものです。


特に、意味的に正しいが不自然な表現や、視覚的特徴のみが一致する「変なラベル」に対して、コサイン類似度がどのように変動するかを定量的に評価します。

## 実験の背景

CLIPは画像とテキストを共通の空間に埋め込むことで強力な汎化性能を持ちますが、特定のキーワードへの過学習や、論理的な否定（"not"など）の理解不足といったバイアスが存在することが知られています。

本実験では、それらの認識境界を明らかにすることを目的にしています。

## 実行環境
計算環境の詳細は以下の通りです。

| 項目 | スペック | 
| -------|-------|
| OS |Windows 11 |
| GPU | NVIDIA GeForce RTX 3070 | 
| Python | 3.10+ |
| Library | "PyTorch, OpenAI-CLIP" |

## セットアップ手順
Anaconda環境での構築を推奨します。

```bash
# 仮想環境の作成
conda create -n clip-eval python=3.10
conda activate clip-eval

# PyTorchのインストール (CUDA環境に合わせる)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# CLIPおよび依存ライブラリのインストール
pip install git+https://github.com/openai/CLIP.git
conda install numpy matplotlib pandas pillow
```

## 使用方法
1. src/config.py に検証したい画像パスとラベルセットを記述します。
2. 以下のコマンドで実験を実行します。

```bash
python src/run_experiment.py
```
実行後、results/ディレクトリに結果のCSVファイルと可視化グラフ(PNG)が生成されます。

## 検証ラベルの設計
本実験では、1つの対象に対して以下のバリエーションのラベルを付与しています。
* Correct: 正解ラベル("a photo of a cat")
* Abstract: 抽象的な説明("a fluffy domestic predetor")
* Adversarial: 視覚的バイアス("a small tiger")
* Negative: 論理的否定("an animal that is not a dog")
* Nonsense: 無意味な文字列("asdfghjkl")

## 著者
* Kay (Tokai Univ.)
* GitHub: [https://github.com/L161803]

## ライセンス
MIT License
