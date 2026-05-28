# debate_wordcloud.py
import os
import json
import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from tqdm import tqdm

BASE_DIR = Path("results/gpt-4o")
DATASETS = ['mmlu', 'truthfulqa', 'medmcqa', 'scalr']
OUTPUT_FORMAT = 'pdf'
OUTPUT_DPI = 300
WORDCLOUD_WIDTH = 1600
WORDCLOUD_HEIGHT = 1000
MAX_WORDS = 150
BACKGROUND_COLOR = 'white'
COLORMAP = 'plasma'
COMPARISON_FIG_SIZE = (16, 12)
COMPARISON_WC_WIDTH = 500
COMPARISON_WC_HEIGHT = 350
COMPARISON_MAX_WORDS = 100
MIN_WORD_LENGTH = 4
SAMPLE_LIMIT = None

ENHANCED_STOPWORDS = set(STOPWORDS).union({
    '的', '了', '是', '在', '和', '有', '这', '那', '也', '不', '与', '而',
    '就', '都', '说', '要', '会', '去', '来', '上', '下', '中', '到', '于',
    '对', '把', '被', '让', '给', '由', '从', '向', '以', '为', '但', '并',
    '或', '如', '等', '还', '只', '可', '能', '过', '吗', '呢', '吧', '啊',
    '我们', '你们', '他们', '它们', '她们', '什么', '怎么', '为什么',
    '哪个', '哪些', '多少', '如何', '怎样', '因为', '所以', '但是',
    '然而', '虽然', '尽管', '即使', '如果', '那么', '这样', '那样',
})


def simple_tokenize(text: str) -> list:
    if not text:
        return []
    
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.lower()
    
    english_words = re.findall(r'[a-z]{4,}', text)
    chinese_words = re.findall(r'[\u4e00-\u9fff]+', text)
    
    all_words = english_words + chinese_words
    return all_words


def clean_and_filter(tokens: list) -> list:
    filtered = []
    for token in tokens:
        token = token.lower().strip()
        
        if len(token) < MIN_WORD_LENGTH:
            continue
        
        if token in ENHANCED_STOPWORDS:
            continue
        
        if len(set(token)) == 1:
            continue
        
        if re.search(r'\d', token):
            continue
        
        filtered.append(token)
    
    return filtered


def load_graph_texts(exp_dir: Path, sample_limit: int = None) -> list:
    texts = []
    graph_files = sorted(exp_dir.glob("graph_*.json"))
    
    if not graph_files:
        return texts
    
    if sample_limit:
        graph_files = graph_files[:sample_limit]
    
    for graph_file in tqdm(graph_files, desc=f"    graph", leave=False):
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            nodes = data.get('nodes', {})
            for node_id, node in nodes.items():
                content = node.get('content', '')
                if content and len(content) > 10:
                    tokens = simple_tokenize(content)
                    filtered_tokens = clean_and_filter(tokens)
                    texts.extend(filtered_tokens)
        except Exception:
            continue
    
    return texts


def load_audit_texts(exp_dir: Path, sample_limit: int = None) -> list:
    texts = []
    audit_dir = exp_dir / "audit_reports"
    if not audit_dir.exists():
        return texts
    
    audit_files = sorted(audit_dir.glob("audit_*.json"))
    
    if sample_limit:
        audit_files = audit_files[:sample_limit]
    
    for audit_file in tqdm(audit_files, desc=f"    audit", leave=False):
        try:
            with open(audit_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            audit_summary = data.get('audit_summary', {})
            
            conclusions = audit_summary.get('conclusions', [])
            for conclusion in conclusions:
                content = conclusion.get('content', '')
                if content and len(content) > 10:
                    tokens = simple_tokenize(content)
                    filtered_tokens = clean_and_filter(tokens)
                    texts.extend(filtered_tokens)
            
            disputes = audit_summary.get('top_dispute_nodes', [])
            for dispute in disputes:
                content = dispute.get('content', '')
                if content and len(content) > 10:
                    tokens = simple_tokenize(content)
                    filtered_tokens = clean_and_filter(tokens)
                    texts.extend(filtered_tokens)
                    
        except Exception:
            continue
    
    return texts


def extract_all_texts(base_dir: Path, dataset_name: str, sample_limit: int = None) -> list:
    all_tokens = []
    dataset_path = base_dir / dataset_name
    
    if not dataset_path.exists():
        return all_tokens
    
    exp_dirs = list(dataset_path.glob("adv_*"))
    
    if not exp_dirs:
        return all_tokens
    
    for exp_dir in exp_dirs:
        graph_tokens = load_graph_texts(exp_dir, sample_limit)
        audit_tokens = load_audit_texts(exp_dir, sample_limit)
        all_tokens.extend(graph_tokens)
        all_tokens.extend(audit_tokens)
    
    return all_tokens


def generate_single_wordcloud(tokens: list, dataset_name: str, output_dir: Path, fmt='pdf'):
    if not tokens:
        return None
    
    freq_counter = Counter(tokens)
    
    wordcloud = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color=BACKGROUND_COLOR,
        max_words=MAX_WORDS,
        colormap=COLORMAP,
        random_state=42,
    ).generate_from_frequencies(freq_counter)
    
    fig, ax = plt.subplots(figsize=(WORDCLOUD_WIDTH/100, WORDCLOUD_HEIGHT/100 + 1), dpi=100)
    
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_xlabel(f'{dataset_name.upper()} - Debate Word Cloud', 
                  fontsize=18, fontweight='bold', labelpad=15)
    
    plt.tight_layout(pad=0)
    
    output_file = output_dir / f"wordcloud_{dataset_name}.{fmt}"
    fig.savefig(output_file, format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return freq_counter


def generate_comparison_wordcloud(all_tokens_dict: dict, output_dir: Path, fmt='pdf'):
    valid_items = [(name, tokens) for name, tokens in all_tokens_dict.items() if tokens]
    n_datasets = len(valid_items)
    
    if n_datasets == 0:
        return
    
    cols = 2
    rows = (n_datasets + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=COMPARISON_FIG_SIZE)
    
    if n_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (dataset_name, tokens) in enumerate(valid_items):
        freq_counter = Counter(tokens)
        
        wordcloud = WordCloud(
            width=COMPARISON_WC_WIDTH,
            height=COMPARISON_WC_HEIGHT,
            background_color='white',
            max_words=COMPARISON_MAX_WORDS,
            colormap=COLORMAP,
            random_state=42
        ).generate_from_frequencies(freq_counter)
        
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].axis('off')
        axes[idx].set_xlabel(dataset_name.upper(), fontsize=14, fontweight='bold', labelpad=10)
    
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    plt.suptitle('Debate Word Clouds Comparison Across Datasets', fontsize=20, y=0.98, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_dir / f"wordcloud_comparison.{fmt}"
    fig.savefig(output_file, format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_all_datasets_wordcloud(all_tokens: list, output_dir: Path, fmt='pdf'):
    if not all_tokens:
        return
    
    freq_counter = Counter(all_tokens)
    
    total_wordcloud = WordCloud(
        width=WORDCLOUD_WIDTH,
        height=WORDCLOUD_HEIGHT,
        background_color=BACKGROUND_COLOR,
        max_words=MAX_WORDS,
        colormap='RdYlBu',
        random_state=42
    ).generate_from_frequencies(freq_counter)
    
    fig, ax = plt.subplots(figsize=(WORDCLOUD_WIDTH/100, WORDCLOUD_HEIGHT/100 + 1), dpi=100)
    
    ax.imshow(total_wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_xlabel('All Datasets Combined - Debate Word Cloud', 
                  fontsize=18, fontweight='bold', labelpad=15)
    
    plt.tight_layout(pad=0)
    
    output_file = output_dir / f"wordcloud_all_datasets.{fmt}"
    fig.savefig(output_file, format=fmt, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    script_dir = Path(__file__).parent.absolute()
    full_base_dir = script_dir / BASE_DIR
    
    if not full_base_dir.exists():
        return
    
    output_dir = full_base_dir / "wordclouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_dataset_tokens = {}
    
    for dataset_name in DATASETS:
        tokens = extract_all_texts(full_base_dir, dataset_name, SAMPLE_LIMIT)
        all_dataset_tokens[dataset_name] = tokens
        
        if tokens:
            generate_single_wordcloud(tokens, dataset_name, output_dir, OUTPUT_FORMAT)
    
    if all_dataset_tokens:
        generate_comparison_wordcloud(all_dataset_tokens, output_dir, OUTPUT_FORMAT)
    
    all_tokens = []
    for tokens in all_dataset_tokens.values():
        all_tokens.extend(tokens)
    
    if all_tokens:
        generate_all_datasets_wordcloud(all_tokens, output_dir, OUTPUT_FORMAT)


if __name__ == "__main__":
    main()