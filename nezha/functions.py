import jieba
from rouge import Rouge
import numpy as np
# 指标名
metric_keys = ['main', 'rouge-1', 'rouge-2', 'rouge-l']

# 计算rouge用
rouge = Rouge()

def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        #把中文内容切成一个字一个字部分
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
    #rouge基于摘要中的n元词的共现来评价摘要，rouge准则由一系列评价方法组成

def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics

def compute_main_metric(source, target, unit='word'):
    """计算主要metric
    """
    return compute_metrics(source, target, unit)['main']

def extract_matching(texts, summaries, start_i=0, start_j=0):
    """在texts中找若干句子，使得它们连起来与summaries尽可能相似
    算法：texts和summaries都分句，然后找出summaries最长的句子，在texts
          中找与之最相似的句子作为匹配，剩下部分递归执行。
    这里采用按顺序抽取的方法，也就是比如递归的结果是原文的第9句对应摘要的第
    4句，那么接下来递归调用的过程中接着找原文中的1~8句和摘要中的1~3句话，
    此算法适用于按照顺序更新的文本
    """
    print('extract_matching')
    if len(texts) == 0 or len(summaries) == 0:
        return []
    i = np.argmax([len(s) for s in summaries])
    print('i = ')
    print(i)
    data1 = [compute_main_metric(t,summaries[i],'char') for t in texts]
    print('data1 = ')
    print(data1)
    j = np.argmax([compute_main_metric(t, summaries[i], 'char') for t in texts])
    print('j = ')
    print(j)
    lm = extract_matching(texts[:j + 1], summaries[:i], start_i, start_j)
    rm = extract_matching(
        texts[j:], summaries[i + 1:], start_i + i + 1, start_j + j
    )
    return lm + [(start_i + i, start_j + j)] + rm

def extract_every_matching(texts, summaries):
    results = []
    for index in range(len(summaries)):
        j = np.argmax([compute_main_metric(t,summaries[index],'char') for t in texts])
        results.append((index,j))
    return results

def sequence_padding(inputs,maxlen,padding = 0):
    length = maxlen
    if len(inputs) > length:
        inputs = inputs[:length]
    outputs = []
    pad_width = (0,length-len(inputs))
    x = np.pad(inputs,pad_width,'constant',constant_values=padding)
    return x
