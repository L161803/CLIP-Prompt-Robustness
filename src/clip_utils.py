# clip_utils.py
import torch
import clip


# ライブラリのロード
def load_clip_model(model_name, device):
    return clip.load(model_name, device=device)


# 類似度の計算
def get_similarity(model, preprocess, image_path, labels, device):
    from PIL import Image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        # L2正規化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 類似度スコア
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    return probs