# Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2504.14194-b31b1b.svg)](https://arxiv.org/abs/2504.14194)
[![Hugging Face Models](https://img.shields.io/badge/🤗HuggingFace-Models-yellow)](https://huggingface.co/opendatalab/)
[![Dataset](https://img.shields.io/badge/🤗HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater)
[![Dataset](https://img.shields.io/badge/OpenDataLab-Dataset-yellow)](https://opendatalab.com/OpenDataLab/SlimPajama-Meta-rater)

*Advancing LLM pre-training efficiency through multi-dimensional data quality assessment*


</div>

## 🎯 Overview

The composition of pre-training datasets for large language models (LLMs) remains largely undisclosed, hindering transparency and efforts to optimize data quality—a critical driver of model performance. **Meta-rater** introduces a groundbreaking multi-dimensional data selection framework that **doubles convergence speed** and improves downstream task performance by **3.23%** compared to random selection.

### 🏆 Key Achievements

- **📈 2x Faster Convergence**: Meta-rater achieves equivalent performance using only 15B tokens compared to 30B tokens with random selection
- **🎯 3.23% Performance Gain**: Significant improvement over random sampling on downstream tasks
- **🔍 Multi-dimensional Quality Assessment**: Novel PRRC framework (Professionalism, Readability, Reasoning, Cleanliness)
- **📊 Scalable Framework**: Benefits persist and increase from 1.3B to 7.2B parameter models
- **🏗️ Comprehensive Dataset**: First fully annotated 627B-token SlimPajama with 25 quality metrics

## 🧠 PRRC Framework

We introduce four novel evaluation dimensions to comprehensively assess data quality:

| Dimension | Description | F1 Score |
|-----------|-------------|----------|
| **🎓 Professionalism** | Degree of expertise and technical knowledge required | 91.57% |
| **📖 Readability** | Ease of understanding and text clarity | 87.47% |
| **🧮 Reasoning** | Complexity of logical thinking and analysis | 89.59% |
| **✨ Cleanliness** | Format quality and noise-free content | 87.88% |

## 🔬 Meta-rater Methodology


Our framework integrates **25 quality scores** across three categories:

1. **Natural Language Quality Signals (11)**: Rule-based measures from RedPajama
2. **Data Importance Scores (3)**: DSIR similarity to Books, Wikipedia, and AutoMathText
3. **Model-based Ratings (11)**: PRRC + QuRating + FineWeb-Edu + WanjuanCC

### Algorithm Overview

```python
# Simplified Meta-rater workflow
for i in range(N_proxy_models):
    weights = generate_random_weights(25)  # Random combination weights
    selected_data = select_top_k(data, weights @ quality_scores)
    proxy_model = train_small_model(selected_data)
    validation_loss = evaluate(proxy_model, validation_set)
    
regression_model = fit_regression(weights, validation_losses)
optimal_weights = find_minimum(regression_model)
final_data = select_top_k(data, optimal_weights @ quality_scores)
```

## 📊 Results

### Main Results (1.3B Models, 30B Tokens)

| Method | General Knowledge | Commonsense Reasoning | Reading Comprehension | **Average** |
|--------|-------------------|----------------------|----------------------|-------------|
| Random Baseline | 52.79 | 43.94 | 30.02 | 43.78 |
| QuRating-Educational Value | 57.66 | 46.72 | 28.10 | 46.16 |
| **Meta-rater (All 25)** | **58.90** | 45.41 | **31.55** | **47.01** |

### Scaling Results

| Model Size | Method | Average Performance | Improvement |
|------------|--------|-------------------|-------------|
| 3.3B | Random | 52.98% | - |
| 3.3B | **Meta-rater** | **54.71%** | **+1.73%** |
| 7.2B | Random | 52.12% | - |
| 7.2B | **Meta-rater** | **55.24%** | **+3.12%** |

> 💡 **Key Insight**: Meta-rater benefits **increase** with model scale, demonstrating that quality data selection becomes more valuable for larger models.

<!-- ## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/opendatalab/Meta-rater.git
cd Meta-rater
pip install -r requirements.txt
```

### Data Rating

Rate your data using our PRRC models:

```python
from meta_rater import PRRCRater

# Initialize rater with all four dimensions
rater = PRRCRater()

# Rate a single text
text = "Your text here..."
scores = rater.rate(text)
print(f"Professionalism: {scores['professionalism']:.2f}")
print(f"Readability: {scores['readability']:.2f}")
print(f"Reasoning: {scores['reasoning']:.2f}")
print(f"Cleanliness: {scores['cleanliness']:.2f}")

# Rate multiple texts
texts = ["Text 1", "Text 2", "Text 3"]
batch_scores = rater.rate_batch(texts)
```

### Data Selection with Meta-rater

```python
from meta_rater import MetaRater
import pandas as pd

# Load your dataset
data = pd.read_json("your_dataset.jsonl", lines=True)

# Initialize Meta-rater with pre-computed optimal weights
meta_rater = MetaRater.from_pretrained("meta-rater-weights")

# Select top-k high-quality examples
selected_indices = meta_rater.select_top_k(
    data, 
    k=1000000,  # Number of examples to select
    quality_scores=["prrc", "qurating", "fineweb", "redpajama", "dsir"]
)

selected_data = data.iloc[selected_indices]
```

### Training Your Own Models -->

<!-- ```python
from meta_rater.training import train_proxy_models, fit_meta_rater

# Step 1: Train proxy models with different weight combinations
proxy_results = train_proxy_models(
    data=your_data,
    n_models=256,
    model_config="configs/proxy_model.yaml"
)

# Step 2: Fit Meta-rater regression model
meta_rater = fit_meta_rater(proxy_results)

# Step 3: Use optimal weights for final data selection
optimal_data = meta_rater.select_final_data(your_data, target_size="30B")
``` -->

## 📦 Available Resources

### 🤖 Pre-trained Models

| Model | Size | Training Tokens | Selection Method | Performance on Downstream Tasks | HF Link |
|-------|------|----------------|------------------|-------------|---------|
| Meta-rater-1.3B | 1.3B | 30B | All 25 scores | 47.01% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-1b-25raters) |
| Meta-rater-3.3B | 3.3B | 100B | All 25 scores | 54.71% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-3b-25raters) |
| Meta-rater-7.2B | 7.2B | 150B | All 25 scores | 55.24% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-7b-25raters) |

### 🎯 PRRC Rating Models

| Model | Dimension | F1 Score on Test set | HF Link |
|-------|------|----------|---------|
| Professionalism | Expertise assessment | 91.57% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-professionalism-rating) |
| Readability | Text clarity rating | 87.47% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-readability-rating) |
| Reasoning | Logic complexity assessment | 89.59% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-reasoning-rating) |
| Cleanliness | Format quality evaluation | 87.88% | [![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/opendatalab/meta-rater-cleanliness-rating) |

### 📊 Datasets

- **Annotated SlimPajama-627B**: [![Dataset](https://img.shields.io/badge/🤗HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater) 
  - 627B tokens with 25 quality scores per document
  - First fully annotated large-scale pre-training dataset
  - Ready for research and production use

- **Top 30B token SlimPajama Subset selected by the Professionalism rater**: [![Dataset](https://img.shields.io/badge/🤗HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater-Professionalism-30B)

- **Top 30B token SlimPajama Subset selected by the Readability rater**: [![Dataset](https://img.shields.io/badge/🤗HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater-Readability-30B)

- **Top 30B token SlimPajama Subset selected by the Reasoning rater**: [![Dataset](https://img.shields.io/badge/🤗HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater-Reasoning-30B)

- **Top 30B token SlimPajama Subset selected by the Cleanliness rater**: [![Dataset](https://img.shields.io/badge/🤗HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/opendatalab/SlimPajama-Meta-rater-Cleanliness-30B)

<!-- ## 📁 Repository Structure

```
Meta-rater/
├── meta_rater/                 # Core library
│   ├── models/                 # PRRC rating models
│   ├── selection/              # Data selection algorithms
│   ├── training/               # Model training utilities
│   └── evaluation/             # Evaluation metrics
├── configs/                    # Configuration files
├── scripts/                    # Training and evaluation scripts
├── notebooks/                  # Example Jupyter notebooks
├── data/                       # Sample data and annotations
├── tests/                      # Unit tests
└── docs/                       # Documentation

```

## 🔧 Advanced Usage

### Custom Quality Score Integration

```python
from meta_rater import MetaRater
from meta_rater.scorers import CustomScorer

# Define your custom quality scorer
class MyCustomScorer(CustomScorer):
    def score(self, text):
        # Your scoring logic here
        return score

# Integrate with Meta-rater
meta_rater = MetaRater()
meta_rater.add_scorer("custom", MyCustomScorer())

# Train with your custom scores
meta_rater.fit(data, include_scores=["prrc", "custom"]) -->
<!-- ``` -->


## 📈 Computational Efficiency

Meta-rater is designed for efficiency:

| Process | FLOPs (×10¹⁹) | Percentage of 1.3B Training |
|---------|---------------|---------------------------|
| Quality Score Rating | 33.0 | 141% |
| Meta-rater Construction | 0.18 | 0.8% |
| **Total Overhead** | **33.2** | **142%** |

> 💡 **Note**: Quality scores are computed once and reused across multiple experiments. For larger models (3.3B+), the overhead becomes negligible (17% for 3.3B training).


## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use Meta-rater in your research, please cite our paper:

```bibtex
@article{zhuang2025meta,
  title={Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models},
  author={Zhuang, Xinlin and Peng, Jiahui and Ma, Ren and Wang, Yinfan and Bai, Tianyi and Wei, Xingjian and Qiu, Jiantao and Zhang, Chi and Qian, Ying and He, Conghui},
  journal={arXiv preprint arXiv:2504.14194},
  year={2025}
}
```

## 🤝 Acknowledgments

- **Shanghai Artificial Intelligence Laboratory** for computational resources
- **InternTrain Team** for pre-training infrastructure support
- **Community contributors** for valuable feedback and improvements

## 📞 Contact

- **Project Lead**: Ren Ma (maren@pjlab.org.cn)
- **Corresponding Author**: Conghui He (heconghui@pjlab.org.cn)
- **Issues**: Please use [GitHub Issues](https://github.com/opendatalab/Meta-rater/issues) for bug reports and feature requests

---

<div align="center">

**⭐ Star us on GitHub if you find Meta-rater useful! ⭐**

Made with ❤️ by the OpenDataLab team

</div> 