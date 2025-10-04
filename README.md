# FinCDM

## 📊 Overview

![FinCDM](/fig/finCDM.png "点击查看大图")

FinCDM (Financial Cognitive Diagnosis Model) is a comprehensive evaluation framework for financial large language models. It moves beyond traditional score-level evaluation by providing knowledge-skill level diagnosis, identifying what financial skills and knowledge models possess or lack.

This project introduces a new paradigm for financial LLM evaluation by enabling interpretable, skill-aware diagnosis that supports more trustworthy and targeted model development.

## 📄 Paper

**From Scores to Skills: A Cognitive Diagnosis Framework for Evaluating Financial Large Language Models**

* 📖 Paper: [Hugging Face Paper Page](https://huggingface.co/papers/2508.13491)
* 📝 arXiv: [arXiv:2508.13491](https://arxiv.org/abs/2508.13491)

### Abstract

We introduce FinCDM, the first cognitive diagnosis evaluation framework tailored for financial LLMs. Unlike existing benchmarks that rely on single aggregate scores, FinCDM evaluates models at the knowledge-skill level, revealing hidden knowledge gaps and identifying under-tested areas such as tax and regulatory reasoning often overlooked by traditional benchmarks.

## 📚 Datasets

We provide two comprehensive datasets for financial LLM evaluation:

### 1. FinCDM-FinEval-KQA

* 🤗  **Hugging Face** : [NextGenWhu/FinCDM-FinEval-KQA](https://huggingface.co/datasets/NextGenWhu/FinCDM-FinEval-KQA)
* **Description** : A knowledge-skill annotated version of FinEval, covering comprehensive financial concepts and skills
* **Features** :
* Fine-grained knowledge labels
* Multi-domain financial questions
* Expert-validated annotations

### 2. FinCDM-CPA-KQA

* 🤗  **Hugging Face** : [NextGenWhu/FinCDM-CPA-KQA](https://huggingface.co/datasets/NextGenWhu/FinCDM-CPA-KQA)
* **Description** : The first cognitively informed financial evaluation dataset derived from the Certified Public Accountant (CPA) examination
* **Features** :
* Comprehensive coverage of real-world accounting and financial skills
* Rigorously annotated by domain experts
* High inter-annotator agreement
* Fine-grained knowledge labels

## 🚀 Features

* [X] Cognitive diagnosis framework for financial LLMs
* [X] Knowledge-skill level evaluation beyond simple scores
* [X] Two comprehensive evaluation datasets (FinEval-KQA and CPA-KQA)
* [ ] Evaluation scripts and tools (coming soon)
* [ ] Model proficiency visualization
* [ ] Skill acquisition pattern analysis
* [ ] Behavioral cluster identification

## 🛠️ Key Innovations

1. **Knowledge-Skill Level Diagnosis** : Unlike traditional benchmarks that provide single scores, FinCDM reveals specific strengths and weaknesses across different financial skills
2. **Comprehensive Coverage** : Tests previously overlooked areas like:

* Tax and regulatory reasoning
* Deferred tax liabilities
* Lease classification
* Regulatory ratios

1. **Model Clustering Analysis** : Identifies latent associations between financial concepts and reveals distinct clusters of models with similar skill acquisition patterns

## 📋 Prerequisites

* Python 3.8+
* Git
* PyTorch >= 1.12.0
* Transformers >= 4.25.0
* NumPy, Pandas, Scikit-learn

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/WHUNextGen/FinCDM.git
cd FinCDM

# Install dependencies (once available)
pip install -r requirements.txt
```

## 📖 Usage

### Loading Datasets

```python
from datasets import load_dataset

# Load FinEval-KQA dataset
fineval_data = load_dataset("NextGenWhu/FinCDM-FinEval-KQA")

# Load CPA-KQA dataset  
cpa_data = load_dataset("NextGenWhu/FinCDM-CPA-KQA")
```

### Running Evaluation

```python
from fincdm import FinCDMEvaluator

# Initialize evaluator
evaluator = FinCDMEvaluator(data_root=".")

# Evaluate your model
results = FinCDMEvaluator().evaluate(
    q_path="",
    a_path="",
)
print(results.metrics)

# Get knowledge-skill diagnosis
diagnosis = evaluator.diagnose(results，export_csv="SK_df.csv")
```

## 📊 Experimental Results

Our extensive experiments on **30+ LLMs** including:

* Proprietary models (GPT-4, GPT-3.5, Claude)
* Open-source models (LLaMA, Mistral, Qwen)
* Domain-specific models (FinGPT, FinMA, FinQwen)

Key findings:

* Reveals hidden knowledge gaps in state-of-the-art models
* Identifies behavioral clusters among different model families
* Uncovers specialization strategies in domain-specific models

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use FinCDM in your research, please cite our paper:

```bibtex
@article{fincdm2024,
  title={From Scores to Skills: A Cognitive Diagnosis Framework for Evaluating Financial Large Language Models},
  author={Kuang, Ziyan and others},
  journal={arXiv preprint arXiv:2508.13491},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

* WHU NextGen Team
* Contributors from Wuhan University

## 🔗 Links

* **GitHub Repository** : [https://github.com/WHUNextGen/FinCDM](https://github.com/WHUNextGen/FinCDM)
* **Paper** : [https://huggingface.co/papers/2412.06264](https://huggingface.co/papers/2412.06264)
* **FinEval-KQA Dataset** : [https://huggingface.co/datasets/NextGenWhu/FinCDM-FinEval-KQA](https://huggingface.co/datasets/NextGenWhu/FinCDM-FinEval-KQA)
* **CPA-KQA Dataset** : [https://huggingface.co/datasets/NextGenWhu/FinCDM-CPA-KQA](https://huggingface.co/datasets/NextGenWhu/FinCDM-CPA-KQA)

## 📧 Contact

For questions and feedback, please:

* Open an issue on GitHub
* Contact the WHU NextGen Team

---

⭐ **Star this repository if you find it helpful!**

🔥 **Check out our datasets on Hugging Face for financial LLM evaluation!**
