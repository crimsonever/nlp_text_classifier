# 说明

这是一个简单的文本分类器，使用方法如下：

1.准备 CSV 文件 data.csv，包含两列：text, label
2.训练并保存模型：
  ```bash
  python nlp_text_classifier.py --mode train --model_out model.joblib
  ```
3.用训练好的模型预测：
  ```bash
  python nlp_text_classifier.py --mode predict --model_in model.joblib --text "待预测文本"
  ```