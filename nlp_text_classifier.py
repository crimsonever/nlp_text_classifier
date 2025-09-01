#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

def identity_func(x):
    return x

# 可选：中文分词（若处理中文文本，安装 jieba 并取消注释下面两行）
try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

def clean_text(text):
    """基础清洗：去除多余空白，URL，特殊字符；中文保留汉字与数字与字母"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    # 去掉 URL
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # 去掉多余空白
    text = re.sub(r'\s+', ' ', text)
    # 保留常见字符（中英文和数字、标点略去）
    # 如果你希望保留标点，可调整下面的正则
    text = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]+', ' ', text)
    return text.strip().lower()

def tokenize_for_tfidf(text):
    """为 TfidfVectorizer 提供分词器：中文用 jieba，英文按空格切分"""
    if _HAS_JIEBA:
        # 中文分词（同时也能处理英文单词）
        return " ".join(jieba.cut(text))
    else:
        # 简单空格分词（英文/已有空格的中文）
        return " ".join(text.split())

def load_data(csv_path, text_col='text', label_col='label', nrows=None):
    df = pd.read_csv(csv_path, nrows=nrows, encoding='utf-8')
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV 必须包含列: {text_col}, {label_col}")
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).map(clean_text)
    return df[text_col].tolist(), df[label_col].tolist()

def build_pipeline(max_features=20000, ngram_range=(1,2)):
    """
    构建 sklearn Pipeline：
      - TfidfVectorizer（自定义 preprocessor + tokenizer）
      - LogisticRegression（多类、可扩展）
    """
    vect = TfidfVectorizer(
        preprocessor=identity_func,   # 代替 lambda
        tokenizer=tokenize_for_tfidf,
        max_features=max_features,
        ngram_range=ngram_range
    )
    clf = LogisticRegression(
        solver='saga',  # 支持大规模、稀疏数据；若数据集小，也可以用 lbfgs
        max_iter=200,
        # multi_class='multinomial',
        C=1.0,
        n_jobs=-1,
        random_state=42
    )
    pipe = Pipeline([
        ('vect', vect),
        ('clf', clf)
    ])
    return pipe

def train_and_evaluate(data_csv, model_out, test_size=0.2, random_state=42):
    print("加载数据...", data_csv)
    X, y = load_data(data_csv)
    print(f"样本数: {len(X)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(set(y))>1 else None
    )
    pipe = build_pipeline()
    print("开始训练...")
    pipe.fit(X_train, y_train)
    print("训练完成，开始评估...")
    y_pred = pipe.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("详细报告:")
    print(classification_report(y_test, y_pred, digits=4))
    # 保存模型
    joblib.dump(pipe, model_out)
    print(f"模型已保存到：{model_out}")

def predict_from_model(model_in, texts):
    if not os.path.exists(model_in):
        raise FileNotFoundError(f"模型文件不存在: {model_in}")
    pipe = joblib.load(model_in)
    # 预处理：clean_text
    texts_clean = [clean_text(t) for t in texts]
    preds = pipe.predict(texts_clean)
    probs = None
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(texts_clean)
    results = []
    for i, t in enumerate(texts):
        r = {"text": t, "pred": preds[i]}
        if probs is not None:
            # top probability
            r["top_prob"] = float(max(probs[i]))
        results.append(r)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'predict'], required=True, help='train 或 predict')
    parser.add_argument('--data', help='训练数据 CSV 路径（train 模式）', default='data/ChnSentiCorp_htl_all.csv')
    parser.add_argument('--model_out', help='训练后保存模型路径（train 模式）', default='model.joblib')
    parser.add_argument('--model_in', help='加载模型路径（predict 模式）', default='model.joblib')
    parser.add_argument('--text', help='要预测的文本（predict 模式）')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.data:
            parser.error("--data 在 train 模式下是必需的")
        train_and_evaluate(args.data, args.model_out, test_size=args.test_size)
    elif args.mode == 'predict':
        if not args.text:
            parser.error("--text 在 predict 模式下是必需的")
        res = predict_from_model(args.model_in, [args.text])
        for r in res:
            print("文本:", r["text"])
            print("预测标签:", r["pred"])
            if "top_prob" in r:
                print("置信度:", r["top_prob"])
            print("-"*30)

if __name__ == '__main__':
    main()
