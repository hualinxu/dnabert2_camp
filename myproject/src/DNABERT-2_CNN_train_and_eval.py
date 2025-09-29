import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef,confusion_matrix
from collections import Counter
from typing import List, Tuple
import logging
import gc
import os
from datetime import datetime
import shutil
from torch.optim import AdamW

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 自定义 Focal Loss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha, gamma, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
#         pt = torch.exp(-ce_loss)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         return focal_loss

# 自定义数据集类
class PromoterDataset(Dataset):
    def __init__(self, sequences: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        seq_6mer = " ".join(sequence[i:i + 6] for i in range(0, len(sequence) - 5))
        encoding = self.tokenizer(
            seq_6mer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_token_type_ids=False
        )
        input_ids = encoding["input_ids"].squeeze()
        if input_ids.eq(0).all():
            logger.error(f"无效 token 化: 序列 {sequence[:10]}... 生成为全 0 input_ids")
            raise ValueError(f"Tokenization failed for sequence {sequence[:10]}...")
        logger.debug(f"Sample {idx}: input_ids={input_ids[:10]}, label={label}")
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 计算类别权重
def compute_class_weights(labels: List[int]) -> torch.Tensor:
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    weights = {k: total / (len(label_counts) * v) for k, v in label_counts.items()}
    return torch.tensor([weights[i] for i in range(2)], dtype=torch.float).to(device)

# 计算评估指标
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, logits[:, 1])
    mcc = matthews_corrcoef(labels, preds)
    # 计算特异性
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    eval_loss = pred.loss if hasattr(pred, "loss") else None
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "mcc": mcc,
        "specificity":specificity
    }
    if eval_loss is not None:
        metrics["eval_loss"] = eval_loss
    return metrics

# CNN-Transformer 混合模型
class CNNTransformerPromoter(nn.Module):
    def __init__(self, dnabert_model_name, num_labels=2, cnn_filters=512, kernel_size=3):
        super().__init__()
        self.dnabert_config = AutoConfig.from_pretrained(dnabert_model_name, trust_remote_code=True, pooler_type="dense")
        logger.info(f"加载 DNABERT 配置: {self.dnabert_config}")
        self.dnabert = AutoModel.from_pretrained(dnabert_model_name, config=self.dnabert_config, trust_remote_code=True)
        logger.info(f"DNABERT 模型加载成功: {dnabert_model_name}")
        self.cnn = nn.Sequential(
            nn.Conv1d(self.dnabert_config.hidden_size, cnn_filters, kernel_size, padding="same"),
            nn.BatchNorm1d(cnn_filters),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters * 3 // 4, kernel_size, padding="same"),
            nn.BatchNorm1d(cnn_filters * 3 // 4),
            nn.ReLU(),
            nn.Conv1d(cnn_filters * 3 // 4, cnn_filters // 2, kernel_size, padding="same"),
            nn.BatchNorm1d(cnn_filters // 2),
            nn.ReLU(),
            nn.Conv1d(cnn_filters // 2, cnn_filters // 4, kernel_size, padding="same"),
            nn.BatchNorm1d(cnn_filters // 4),
            nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(cnn_filters // 4, num_heads=4)
        self.fusion = nn.Linear(cnn_filters // 4 + self.dnabert_config.hidden_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.norm = nn.LayerNorm(128)
        self.classifier = nn.Linear(128, num_labels)
        self.class_weights = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        dnabert_outputs = self.dnabert(input_ids, attention_mask=attention_mask)
        sequence_output = dnabert_outputs[0]
        transformer_pooled = dnabert_outputs[1] if len(dnabert_outputs) > 1 else sequence_output[:, 0]
        logger.debug(f"Transformer pooled shape: {transformer_pooled.shape}")
        cnn_input = sequence_output.transpose(1, 2)
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.transpose(1, 2)
        attn_output, _ = self.attention(cnn_output, cnn_output, cnn_output)
        cnn_pooled = attn_output.mean(dim=1)
        fused = torch.cat([transformer_pooled, cnn_pooled], dim=-1)
        fused = self.fusion(fused)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=dnabert_outputs[1] if len(dnabert_outputs) > 1 else None,
            attentions=dnabert_outputs[2] if len(dnabert_outputs) > 2 else None
        )

    def save_custom_weights(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.dnabert.save_pretrained(save_dir)
        self.dnabert_config.save_pretrained(save_dir)
        custom_weights = {
            'cnn': self.cnn.state_dict(),
            'fusion': self.fusion.state_dict(),
            'classifier': self.classifier.state_dict(),
            'dropout': self.dropout.state_dict(),
            'norm': self.norm.state_dict(),
            'attention': self.attention.state_dict()
        }
        torch.save(custom_weights, os.path.join(save_dir, "custom_weights.pth"))
        logger.info(f"自定义模型权重保存至: {save_dir}")

    def load_custom_weights(self, save_dir):
        self.dnabert = AutoModel.from_pretrained(save_dir, trust_remote_code=True)
        custom_weights = torch.load(os.path.join(save_dir, "custom_weights.pth"), map_location=device)
        self.cnn.load_state_dict(custom_weights['cnn'])
        self.fusion.load_state_dict(custom_weights['fusion'])
        self.classifier.load_state_dict(custom_weights['classifier'])
        self.dropout.load_state_dict(custom_weights['dropout'])
        self.norm.load_state_dict(custom_weights['norm'])
        self.attention.load_state_dict(custom_weights['attention'])
        logger.info(f"自定义模型权重加载自: {save_dir}")

# 预处理数据（无增强）
def preprocess_data(data_file: str) -> Tuple[List[str], List[int], List[str]]:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据文件 {data_file} 不存在")
    df = pd.read_csv(data_file)
    sequences = []
    labels = []
    sigma_types = []
    seen_sequences = set()

    for _, row in df.iterrows():
        seq = row["Sequence"].upper().strip()
        try:
            label = int(row["label"])
        except (ValueError, TypeError):
            logger.warning(f"跳过无效标签: {row['label']}")
            continue
        if not all(base in "ATCG" for base in seq) or len(seq) != 81:
            logger.warning(f"跳过无效序列: {seq[:10]}... (长度: {len(seq)})")
            continue
        sigma_type = row.get("sigma_type", "none")

        if seq not in seen_sequences:
            sequences.append(seq)
            labels.append(label)
            sigma_types.append(sigma_type)
            seen_sequences.add(seq)

    logger.info(f"从 {data_file} 加载了 {len(sequences)} 条序列，正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")
    logger.info(f"正负比例: {sum(labels) / (len(labels) - sum(labels)):.2f}:1")
    return sequences, labels, sigma_types

# 主训练函数
def train_and_evaluate(data_file: str = "../../Data/original_dataset/original_dataset(8220).csv"):
    logger.info(f"加载数据集: {data_file}")
    sequences, labels, sigma_types = preprocess_data(data_file)
    class_weights = compute_class_weights(labels)
    logger.info(f"类别权重: {class_weights.tolist()}")

    train_val_seqs, test_seqs, train_val_labels, test_labels = train_test_split(
        sequences, labels, test_size=0.1, stratify=labels, random_state=42
    )
    logger.info(f"训练+验证集: {len(train_val_seqs)} 条，测试集: {len(test_seqs)} 条")
    logger.info(f"正样本比例: {sum(train_val_labels) / len(train_val_labels):.4f} (训练+验证), "
                f"{sum(test_labels) / len(test_labels):.4f} (测试)")

    model_path = "../model"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    fold_sizes = []
    best_roc_auc = -float('inf')
    best_model_dir = "../SaveModel/best_cnn_dnabert2_model"

    # 累计学习：模型和优化器在循环外初始化
    model = CNNTransformerPromoter(model_path, num_labels=2).to(device)
    model.class_weights = class_weights


    explicit_args = {
        "output_dir": "./results_fold_0",
        "num_train_epochs": 15,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.05,
        "warmup_steps": 300,
        "logging_steps": 10,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "roc_auc",
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "fp16": False,
        "logging_dir": "./logs",
        "greater_is_better": True,
        "max_grad_norm": 0.8
    }
    training_args = TrainingArguments(**explicit_args)
    optimizer = AdamW(model.parameters(), lr=explicit_args["learning_rate"])
    with open("../Log/DNABERT-2_CNN_log.txt", "a") as f:
        f.write(f"\n=== Training Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Data File: {data_file}\n")
        f.write("Training Arguments:\n")
        f.write("save to cnn_model(no enhance)\n")
        for arg, value in explicit_args.items():
            f.write(f"  {arg}: {value}\n")
        f.write(f"Class Weights: {class_weights.tolist()}\n")
        f.write("\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_seqs, train_val_labels)):
        logger.info(f"训练第 {fold + 1}/5折")

        train_seqs = [train_val_seqs[i] for i in train_idx]
        train_labels = [train_val_labels[i] for i in train_idx]
        val_seqs = [train_val_seqs[i] for i in val_idx]
        val_labels = [train_val_labels[i] for i in val_idx]

        logger.info(f"第 {fold + 1} 折训练集正样本: {sum(train_labels)}, 负样本: {len(train_labels) - sum(train_labels)}")
        logger.info(f"第 {fold + 1} 折验证集正样本: {sum(val_labels)}, 负样本: {len(val_labels) - sum(val_labels)}")

        train_dataset = PromoterDataset(train_seqs, train_labels, tokenizer)
        val_dataset = PromoterDataset(val_seqs, val_labels, tokenizer)

        training_args.output_dir = f"./results_fold_{fold}"

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"检测到无效损失: {loss.item()}")
                    loss = torch.tensor(0.0, requires_grad=True).to(device)
                logger.debug(f"损失值: {loss.item():.4f}, logits 样本: {outputs.logits[:2]}")
                return (loss, outputs) if return_outputs else loss

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=10,
                early_stopping_threshold=0.01
            )]
        )

        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics)
        fold_sizes.append(len(val_idx))
        logger.info(f"第 {fold + 1} 折指标: {metrics}, 样本量: {len(val_idx)}")

        current_roc_auc = metrics.get("eval_roc_auc", -float('inf'))
        if current_roc_auc > best_roc_auc:
            best_roc_auc = current_roc_auc
            logger.info(f"发现更好的模型在第 {fold + 1} 折，ROC AUC: {current_roc_auc:.4f}")
            if os.path.exists(best_model_dir):
                shutil.rmtree(best_model_dir)
            os.makedirs(best_model_dir, exist_ok=True)
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            logger.info(f"最佳模型保存至: {best_model_dir}")

        del trainer, train_dataset, val_dataset
        torch.cuda.empty_cache()
        gc.collect()

    avg_metrics = {}
    for key in fold_metrics[0].keys():
        values = [m[key] for m in fold_metrics]
        weights = [s / sum(fold_sizes) for s in fold_sizes]
        avg_metrics[key] = np.average(values, weights=weights)
        logger.info(f"{key} 每折值: {values}")
    logger.info(f"交叉验证加权平均指标: {avg_metrics}")

    with open("../Log/DNABERT-2_CNN_log.txt", "a") as f:
        f.write(f"\n=== Cross-Validation Summary at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write("Average Metrics Across Folds:\n")
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics]
            f.write(f"  {key} 每折值: {values}\n")
            f.write(f"  {key} 加权平均: {avg_metrics[key]:.4f}\n")
        f.write("\n")

    # 测试集评估
    test_dataset = PromoterDataset(test_seqs, test_labels, tokenizer)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    test_metrics = trainer.evaluate()
    logger.info(f"测试集样本量: {len(test_seqs)}, 正样本: {sum(test_labels)}, 负样本: {len(test_labels) - sum(test_labels)}")
    logger.info(f"测试集指标: {test_metrics}")

    with open("../Log/DNABERT-2_CNN_log.txt", "a") as f:
        f.write(f"\n=== Test Set Evaluation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Test Set Size: {len(test_seqs)}, Positive: {sum(test_labels)}, Negative: {len(test_labels) - sum(test_labels)}\n")
        f.write("Test Metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")

    return model, tokenizer

# 预测新序列
def predict_promoter(sequence: str, model, tokenizer) -> Tuple[float, bool]:
    sequence = sequence.upper().strip()
    if not all(base in "ATCG" for base in sequence) or len(sequence) != 81:
        raise ValueError(f"无效序列：必须是 80 bp，仅包含 A、T、C、G (当前长度: {len(sequence)})")
    seq_6mer = " ".join(sequence[i:i + 6] for i in range(0, len(sequence) - 5))
    encoding = tokenizer(
        seq_6mer,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        promoter_prob = probs[0][1].item()
        is_promoter = promoter_prob > 0.5
    return promoter_prob, is_promoter

if __name__ == "__main__":
    model, tokenizer = train_and_evaluate(data_file="../../Data/original_dataset/original_dataset(8720).csv")
    test_sequence = "TCGCACGGGTGGATAAGCGTTTACAGTTTTCGCAAGCTCGTAAAAGCAGTACAGTGCACCGTAAGAAAATTACAAGTATAC"
    prob, is_promoter = predict_promoter(test_sequence, model, tokenizer)
    logger.info(f"测试序列预测: 启动子概率 = {prob:.4f}, 是否为启动子 = {is_promoter}")