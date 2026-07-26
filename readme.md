# Debate2Graph: Structured Argumentation for Collaborative Reasoning in Large Language Models

A framework for simulating multi-agent debates with argumentation graph construction, audit visualization, and quality analysis. Supports multiple datasets and LLM backends.

## Features

- **Multi-Agent Debate Simulation**: Run debates with configurable numbers of agents and rounds
- **Argumentation Graph Construction**: Automatically extract claims, evidence, assumptions, and conclusions from agent responses
- **Relation Identification**: Detect supports, attacks, and citations between arguments
- **Audit & Visualization**: Generate audit reports, HTML summaries, and statistical analyses
- **Quality Metrics**: Compute convergence ratios, attack densities, support densities, and other debate quality metrics
- **Cross-Model Analysis**: Aggregate results across different models and datasets
- **Token Efficiency Tracking**: Monitor token consumption and API costs
- **Word Cloud Generation**: Visualize debate content across datasets
- **Statistical Analysis**: Compute confidence intervals and perform significance tests for comparative evaluation

## Supported Datasets

| Dataset | Domain | Format |
|---------|--------|--------|
| MMLU | General knowledge | Multiple choice |
| MATH | Mathematical reasoning | LaTeX problems |
| TruthfulQA | Factual accuracy | Multiple choice |
| MedMCQA | Medical knowledge | Multiple choice |
| SCALR | Legal reasoning | Multiple choice |
| MQuake | Multi-hop reasoning | Question answering |
| Musique | Multi-hop reasoning | Question answering |
| Chess | Board game reasoning | Move prediction |

## Supported Models

Any OpenAI-compatible API endpoint, including:
- GPT-4o / GPT-4 / GPT-3.5
- DeepSeek-R1
- Qwen 3.5 (27B)
- Custom models via compatible APIs

## Installation

```bash
# Install dependencies
pip install openai tiktoken pandas matplotlib seaborn wordcloud tqdm datasets scipy
```

## Configuration

Set your API credentials as environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="your-api-endpoint"  
```

## Usage

### Run a Debate Simulation

```bash
python moderate_main.py \
    --dataset mmlu \
    --n_samples 50 \
    --n_agents 3 \
    --n_rounds 3 \
    --n_reps 1 \
    --output_dir results/gpt-4o \
    --model_name gpt-4o
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset name | `mmlu` |
| `--input_file` | Custom input file path | `None` |
| `--n_samples` | Number of samples to process | `50` |
| `--n_agents` | Number of debating agents | `3` |
| `--n_rounds` | Number of debate rounds | `3` |
| `--n_reps` | Number of repetitions | `1` |
| `--output_dir` | Output directory | `results/gpt-4o` |
| `--model_name` | Model name | `gpt-4o` |

### Analyze Debate Quality

```bash
# Single experiment analysis
python plot.py --single results/gpt-4o/mmlu/adv_50_3_3

# Batch analysis across all experiments
python plot.py --base_dir results/gpt-4o

# Generate case study for specific sample
python plot.py --base_dir results/gpt-4o --sample_id 25
```

### Cross-Model Aggregated Analysis

```bash
python plot_aggregated.py \
    --base_dir results \
    --models gpt-4o deepseek-r1 qwen3.5-27b \
    --datasets mmlu truthfulqa medmcqa
```

### Token Consumption Analysis

```bash
# Batch analysis
python count_tokens.py --base_dir results/gpt-4o

# Single experiment
python count_tokens.py --single results/gpt-4o/mmlu/adv_50_3_3
```

### Efficiency Metrics

```bash
python efficiency_metrics.py --base_dir results/qwen3.5-27b
```

### Statistical Analysis

Compute 95% confidence intervals for debate quality metrics:

```bash
python compute_confidence_intervals.py --base_dir results/gpt-4o

# Analyze specific dataset
python compute_confidence_intervals.py --base_dir results/gpt-4o --dataset mmlu

# Specify output directory
python compute_confidence_intervals.py --base_dir results/gpt-4o --output_dir results/analysis
```

Extract multiple-choice answers from LLM responses:

```python
from answer_parser import extract_choice_answer

response = "The correct answer is (A)"
answer = extract_choice_answer(response)
print(answer)  # 'A'
```

### Generate Word Clouds

```bash
# Configure datasets in debate_wordcloud.py then run
python debate_wordcloud.py
```

## Output Structure

```
results/
└── {model_name}/
    └── {dataset}/
        └── adv_{samples}_{agents}_{rounds}/
            ├── adv_*.jsonl              # Raw debate data
            ├── graph_*.json             # Argumentation graphs
            ├── audit_reports/
            │   └── audit_*.json         # Audit reports
            ├── analysis/
            │   ├── statistics_report.json
            │   ├── debate_analysis_results.csv
            │   ├── distribution_plots.pdf
            │   ├── round_progression.pdf
            │   ├── agent_contributions.pdf
            │   ├── correlation_heatmap.pdf
            │   └── case_study_sample_*.txt
            └── wordclouds/
                ├── wordcloud_*.pdf
                ├── wordcloud_comparison.pdf
                └── wordcloud_all_datasets.pdf
```

## Output Files Description

| File | Description |
|------|-------------|
| `adv_*.jsonl` | Complete debate records with agent responses and metadata |
| `graph_*.json` | Argumentation graph with nodes (claims/evidence) and edges (supports/attacks) |
| `audit_*.json` | Audit summary with conclusions, disputes, and consensus points |
| `statistics_report.json` | Aggregated statistics across all samples |
| `distribution_plots.pdf` | Histograms of nodes, edges, convergence, and attack density |
| `round_progression.pdf` | Debate depth progression across rounds |
| `agent_contributions.pdf` | Agent contribution distributions |
| `correlation_heatmap.pdf` | Metric correlation analysis |
| `case_study_sample_*.txt` | Detailed analysis of a representative sample |

## Utility Modules

| Module | Description |
|--------|-------------|
| `answer_parser.py` | Extract multiple-choice answers from LLM responses |
| `compute_confidence_intervals.py` | Compute 95% confidence intervals via bootstrap |
| `significance_test.py` | Perform paired t-test and Wilcoxon signed-rank test |

## Citation

Coming soon.

**Main Updates:**

1. **Added Features Entry**: Added "Statistical Analysis"
2. **Added Utility Modules Table**: Listed 3 new utility modules and their functions
3. **Added Statistical Analysis Usage Examples**:
- Usage of `compute_confidence_intervals.py`
- Usage of `significance_test.py`
- Usage of `answer_parser.py`
4. **Updated Dependencies**: Added `scipy` to the installation command
