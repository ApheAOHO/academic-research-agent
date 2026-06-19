# evaluator.py
"""
算法评测模块：通过句子向量相似度计算研究报告的 Precision 和 Recall
- Precision: 报告中的句子与参考材料（搜索结果+抓取内容）一致的比例
- Recall: 参考材料中的重要信息被报告覆盖的比例
"""
import re
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# 全局编码器（复用，避免重复加载）
_encoder = None
_DEFAULT_THRESHOLD = 0.7  # 相似度阈值，可调整


def _get_encoder():
    """懒加载句子编码模型（使用轻量级 all-MiniLM-L6-v2）"""
    global _encoder
    if _encoder is None:
        print("🧠 加载句子编码器用于 Precision/Recall 评测...")
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return _encoder


def split_sentences(text: str, min_length: int = 15) -> List[str]:
    """
    将文本分句（按中文/英文标点）
    - min_length: 过滤过短的句子（字符数）
    """
    # 按句子分隔符切分
    delimiter_pattern = r'[。!?！？\n]+'
    raw_sents = re.split(delimiter_pattern, text)
    # 清理并过滤短句
    sentences = []
    for s in raw_sents:
        s = s.strip()
        if len(s) >= min_length:
            sentences.append(s)
    return sentences


def compute_precision_recall(
    report: str,
    reference_corpus: str,
    threshold: float = _DEFAULT_THRESHOLD,
    use_embedding_cache: bool = True
) -> Dict:
    """
    计算 Precision 和 Recall

    参数:
        report: 最终生成的学术报告
        reference_corpus: 本次研究中收集的所有参考文本（搜索结果 + 抓取内容）
        threshold: 相似度阈值（默认 0.7），高于此值认为匹配
        use_embedding_cache: 是否缓存编码结果（对同一模型调用有效）

    返回:
        {
            "precision": float,   # 0~1
            "recall": float,      # 0~1
            "details": {
                "report_sentences": int,
                "ref_sentences": int,
                "correct_report_sents": int,
                "covered_ref_sents": int,
                "threshold": float
            }
        }
    """
    if not report or not reference_corpus:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "details": {
                "report_sentences": 0,
                "ref_sentences": 0,
                "correct_report_sents": 0,
                "covered_ref_sents": 0,
                "threshold": threshold,
                "error": "Empty report or reference corpus"
            }
        }

    encoder = _get_encoder()

    # 分句
    report_sents = split_sentences(report)
    ref_sents = split_sentences(reference_corpus)

    if not report_sents or not ref_sents:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "details": {
                "report_sentences": len(report_sents),
                "ref_sentences": len(ref_sents),
                "correct_report_sents": 0,
                "covered_ref_sents": 0,
                "threshold": threshold,
                "error": "No valid sentences after splitting"
            }
        }

    # 编码所有句子（编码器已经归一化，可直接点积得余弦相似度）
    report_emb = encoder.encode(report_sents, convert_to_tensor=True)
    ref_emb = encoder.encode(ref_sents, convert_to_tensor=True)

    # 计算相似度矩阵 (report_sents x ref_sents)
    sim_matrix = report_emb @ ref_emb.T
    sim_matrix = sim_matrix.cpu().numpy()

    # Precision: 对每个报告句子，最大相似度 > 阈值则正确
    max_sim_for_report = np.max(sim_matrix, axis=1)
    correct_report_sents = np.sum(max_sim_for_report >= threshold)
    precision = correct_report_sents / len(report_sents) if report_sents else 0.0

    # Recall: 对每个参考句子，最大相似度 > 阈值则被覆盖
    max_sim_for_ref = np.max(sim_matrix, axis=0)
    covered_ref_sents = np.sum(max_sim_for_ref >= threshold)
    recall = covered_ref_sents / len(ref_sents) if ref_sents else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "details": {
            "report_sentences": len(report_sents),
            "ref_sentences": len(ref_sents),
            "correct_report_sents": int(correct_report_sents),
            "covered_ref_sents": int(covered_ref_sents),
            "threshold": threshold
        }
    }


def compute_f1(precision: float, recall: float) -> float:
    """计算 F1 分数（调和平均）"""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# 快捷调用：返回包含 F1 的完整结果
def evaluate_full(report: str, reference_corpus: str, threshold: float = 0.7) -> Dict:
    """返回包含 Precision, Recall, F1 的完整评测结果"""
    pr = compute_precision_recall(report, reference_corpus, threshold)
    pr["f1"] = compute_f1(pr["precision"], pr["recall"])
    return pr