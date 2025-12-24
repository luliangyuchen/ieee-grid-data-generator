# IEEE Grid Data Generator

`ieee-grid-data-generator` is a modular data generation framework for power system learning tasks.  
Its primary goal is to convert raw power system cases from different simulation tools (e.g., PyPower, PSASP) into a **unified, model-agnostic dataset scheme**, which can be further transformed into model-ready representations such as graph datasets or vectorized datasets.

This repository focuses on producing **clean, consistent, and reusable scheme-level datasets**, rather than model-specific data structures.

---

## Motivation

Learning-based methods for power system analysis often tightly couple:
- raw simulator data formats,
- preprocessing and cleaning logic,
- and model-specific input representations.

Such coupling significantly limits code reuse and makes it difficult to extend pipelines to new data sources or new model architectures.

This project addresses this issue by explicitly separating:
1. **Source adaptation** (tool-specific),
2. **Scheme-level dataset construction** (tool- and model-agnostic).

---

## Project Scope

This repository is responsible for:
- Adapting raw cases from different simulators into a unified scheme
- Constructing scheme-compliant datasets in PyTorch format
- Providing a stable intermediate representation for downstream model pipelines

This repository does **not**:
- implement neural network models,
- enforce a specific graph framework (PyG, DGL, etc.),
- perform model-side batching, padding, or normalization.

---

## Dataset Scheme Overview

The core output of this project is a **scheme-compliant dataset** represented in PyTorch format.

The dataset serves as an **intermediate representation (IR)** of power system samples and is intentionally designed to be:
- model-agnostic,
- source-agnostic,
- reusable across different learning tasks and architectures.

---

## Dataset Format

The dataset is represented as a Python dictionary:

```python
dataset: Dict[str, List[torch.Tensor]]

