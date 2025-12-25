import torch
import logging
import transformers
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from typing import Tuple, List
import argparse
import transformers

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检查 GPU 可用性
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")


# 加载模型类
class CNNTransformerPromoter(torch.nn.Module):
    def __init__(self, dnabert_model_name, num_labels=2, cnn_filters=64, kernel_size=3):
        super().__init__()
        self.dnabert = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True)
        self.dnabert_config = AutoModel.from_pretrained(dnabert_model_name, trust_remote_code=True).config
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(self.dnabert_config.hidden_size, cnn_filters, kernel_size, padding="same"),
            torch.nn.ReLU(),
            torch.nn.Conv1d(cnn_filters, cnn_filters // 2, kernel_size, padding="same"),
            torch.nn.ReLU()
        )
        self.fusion = torch.nn.Linear(cnn_filters // 2 + self.dnabert_config.hidden_size, 128)
        self.dropout = torch.nn.Dropout(0.4)
        self.norm = torch.nn.LayerNorm(128)
        self.classifier = torch.nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        dnabert_outputs = self.dnabert(input_ids, attention_mask=attention_mask)
        sequence_output = dnabert_outputs[0]
        cnn_input = sequence_output.transpose(1, 2)
        cnn_output = self.cnn(cnn_input)
        cnn_pooled = cnn_output.mean(dim=2)
        transformer_pooled = sequence_output[:, 0]
        fused = torch.cat([transformer_pooled, cnn_pooled], dim=-1)
        fused = self.fusion(fused)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        logits = self.classifier(fused)

        loss = None
        if labels is not None:
            class_weights = torch.tensor([1.0, 1.0]).to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=dnabert_outputs[1] if len(dnabert_outputs) > 1 else None,
            attentions=dnabert_outputs[2] if len(dnabert_outputs) > 2 else None
        )

    def load_custom_weights(self, save_dir):
        self.dnabert = AutoModel.from_pretrained(save_dir, trust_remote_code=True).to(device)
        custom_weights = torch.load(os.path.join(save_dir, "custom_weights.pth"), map_location=device)
        self.cnn.load_state_dict(custom_weights['cnn'])
        self.fusion.load_state_dict(custom_weights['fusion'])
        self.classifier.load_state_dict(custom_weights['classifier'])
        self.dropout.load_state_dict(custom_weights['dropout'])
        self.norm.load_state_dict(custom_weights['norm'])
        self.to(device)
        logger.info(f"自定义模型权重加载自: {save_dir}, 模型设备: {next(self.parameters()).device}")


# 预测函数
def predict_promoter(sequence: str, model, tokenizer, max_length: int = 128) -> Tuple[float, bool]:
    sequence = sequence.upper().strip()
    if not all(base in "ATCG" for base in sequence) or len(sequence) != 81:
        raise ValueError("无效序列：必须是 81 bp，仅包含 A、T、C、G")
    seq_6mer = " ".join(sequence[i:i + 6] for i in range(0, len(sequence) - 5))
    encoding = tokenizer(
        seq_6mer,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False
    )
    logger.debug(f"Tokenizer output: {encoding.keys()}")
    for k, v in encoding.items():
        logger.debug(f"{k} initial device: {v.device}")
    encoding = {
        "input_ids": encoding["input_ids"].to(device),
        "attention_mask": encoding["attention_mask"].to(device)
    }
    logger.debug(
        f"Input tensors device: input_ids={encoding['input_ids'].device}, attention_mask={encoding['attention_mask'].device}")
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1)
        promoter_prob = probs[0][1].item()
        is_promoter = promoter_prob > 0.5
    return promoter_prob, is_promoter


# 批量预测函数
def predict_batch(sequences: List[str], model, tokenizer, max_length: int = 128) -> List[Tuple[str, float, bool]]:
    results = []
    for seq in sequences:
        try:
            prob, is_promoter = predict_promoter(seq, model, tokenizer, max_length)
            results.append((seq, prob, is_promoter))
            logger.info(f"序列 {seq[:10]}...: 启动子概率 = {prob:.4f}, 是否为启动子 = {is_promoter}")
        except ValueError as e:
            logger.warning(f"跳过无效序列 {seq[:10]}...: {str(e)}")
    return results


# 主函数
def main():
    parser = argparse.ArgumentParser(description="使用 DNABERT-2 模型进行启动子预测")
    parser.add_argument("--model_dir", default="./best_cnn_dnabert2_model", help="模型保存目录")
    parser.add_argument("--input_file", default=None, help="输入 CSV 文件路径（包含 Sequence 列）")
    parser.add_argument("--sequence", default=None, help="单条 DNA 序列")
    parser.add_argument("--output_file", default="predictions.csv", help="预测结果输出文件")
    args = parser.parse_args()

    # 清空 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 加载 tokenizer 和模型
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
        model = CNNTransformerPromoter(args.model_dir, num_labels=2).to(device)
        model.load_custom_weights(args.model_dir)
        # 检查模型参数设备
        for name, param in model.named_parameters():
            if param.device != device:
                logger.error(f"参数 {name} 在 {param.device}，应为 {device}")
                return
        logger.info(f"所有模型参数在 {device}")
    except Exception as e:
        logger.error(f"加载模型或 tokenizer 失败: {str(e)}")
        return

    # 处理输入
    if args.input_file:
        if not os.path.exists(args.input_file):
            logger.error(f"输入文件 {args.input_file} 不存在")
            return
        df = pd.read_csv(args.input_file)
        if "Sequence" not in df.columns:
            logger.error("输入 CSV 必须包含 'Sequence' 列")
            return
        sequences = df["Sequence"].tolist()
        results = predict_batch(sequences, model, tokenizer)

        # 保存结果
        output_df = pd.DataFrame(results, columns=["Sequence", "Promoter_Probability", "Is_Promoter"])
        output_df.to_csv(args.output_file, index=False)
        logger.info(f"预测结果保存至: {args.output_file}")

    elif args.sequence:
        try:
            prob, is_promoter = predict_promoter(args.sequence, model, tokenizer)
            logger.info(f"序列 {args.sequence[:10]}...: 启动子概率 = {prob:.4f}, 是否为启动子 = {is_promoter}")
            pd.DataFrame([{
                "Sequence": args.sequence,
                "Promoter_Probability": prob,
                "Is_Promoter": is_promoter
            }]).to_csv(args.output_file, index=False)
            logger.info(f"预测结果保存至: {args.output_file}")
        except ValueError as e:
            logger.error(f"预测失败: {str(e)}")
    else:
        logger.error("必须提供 --input_file 或 --sequence 参数")
        return


if __name__ == "__main__":
    main()
