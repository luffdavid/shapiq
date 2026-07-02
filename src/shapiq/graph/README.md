<div align="center">

# GraphSHAP-IQ Guide

**Group E: Practical Toolbox for Trustworthy Machine Learning and xAI, LMU**

</div>

> This document combines two goals:
> - Project understanding: why GraphSHAP-IQ exists, how it works, and how it fits into shapiq.
> - Review efficiency: a strict reading order and concrete checkpoints for a technical review.

---

## 📚 Table Of Contents
- [Purpose And Motivation](#purpose-and-motivation)
- [High-Level Project Overview](#high-level-project-overview)
- [Core Concepts And Terminology](#core-concepts-and-terminology)
- [Subpackage Architecture & File-by-File Overview](#subpackage-architecture--file-by-file-overview)
- [Package Structure](#package-structure)
- [Usage Notes](#usage-notes)
- [Testing And Validation & Tests to Inspect](#testing-and-validation--tests-to-inspect)
- [Quick Start: Reading Order for Reviewers](#quick-start-reading-order-for-reviewers)
- [Detailed File Descriptions](#detailed-file-descriptions)
- [Demo & Validation Artifacts](#demo--validation-artifacts)
- [References](#references)

---

## Purpose And Motivation

GraphSHAP-IQ computes exact any-order Shapley interaction values for Graph Neural Networks (GNNs).

The core motivation is message-passing locality:

- After L layers, a node representation depends only on its L-hop receptive field.
- Interactions that cannot co-occur inside relevant receptive fields are zero.
- Coalition evaluation can therefore be pruned from brute force $O(2^n)$ toward $O(n \cdot 2^r)$, where $r$ is the largest receptive field size.

In shapiq, this is the graph counterpart to TreeSHAP-IQ: exact, structure-aware interaction explanations.

---

## High-Level Project Overview

GraphSHAP-IQ in this repository is implemented as a graph subpackage with three main layers:

1. GraphGame: wraps a PyTorch Geometric graph/model as a cooperative game over nodes.
2. GraphSHAPIQ: computes Möbius coefficients on locality-pruned coalitions and converts to interaction indices.
3. GraphExplainer: user-facing integration with shapiq's Explainer framework.

An optional approximation route (LShapley) is provided for budget-aware scenarios.

---

## Subpackage Architecture & File-by-File Overview

🏗️ Current implementation in [src/shapiq/graph](src/shapiq/graph):

- [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py): public subpackage exports (GraphGame, GraphExplainer, GraphSHAPIQ).
- [src/shapiq/graph/base.py](src/shapiq/graph/base.py): GraphGame.
- [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py): GraphSHAPIQ core algorithm.
- [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py): GraphExplainer integration.
- [src/shapiq/graph/l_shapley.py](src/shapiq/graph/l_shapley.py): LShapley approximation.
- [src/shapiq/graph/demo](src/shapiq/graph/demo): notebooks, models, and generated artifacts.

| File | Role |
| --- | --- |
| [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py) | Subpackage API |
| [src/shapiq/graph/base.py](src/shapiq/graph/base.py) | GraphGame |
| [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py) | Exact algorithm |
| [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py) | User-facing explainer |
| [src/shapiq/graph/l_shapley.py](src/shapiq/graph/l_shapley.py) | Approximation |

> [!IMPORTANT]
> GraphGame currently lives in [src/shapiq/graph/base.py](src/shapiq/graph/base.py), not in a separate game module.

---

## Package Structure

📂 Quick tree view of the graph subpackage:

```text
src/shapiq/graph/
├── __init__.py
├── base.py
├── explainer.py
├── graphshapiq.py
├── l_shapley.py
└── demo/
    ├── benzene.ipynb
    ├── b-xaic_training.ipynb
    ├── b-xaic_explainers.ipynb
    ├── benzene_data/
    ├── models/
    ├── results/
    └── training_plots/
```

---

## Usage Notes

### Minimal Example: 

```python
from shapiq.graph import GraphExplainer

explainer = GraphExplainer(
    model=model,
   baseline_strategy="zeros",
   l_shapley_max_budget=max_budget,
   max_order=max_order,
   task="classification", 
)

interaction_values = explainer.explain(
   x_graph,
   max_interaction_size=max_order,
   max_subset_size=max_order,
   l_shapley=l_shapley,
   efficiency_routine=False,
)
```
---

## Testing And Validation & Tests to Inspect

Graph-specific tests are in [tests/shapiq/graph](tests/shapiq/graph):

- [tests/shapiq/graph/conftest.py](tests/shapiq/graph/conftest.py): model and graph fixtures (GCN/GIN/GAT and edge-case graph topologies).
- [tests/shapiq/graph/test_graph_game.py](tests/shapiq/graph/test_graph_game.py): masking, baselines, value_function behavior, input checks.
- [tests/shapiq/graph/test_graphshapiq.py](tests/shapiq/graph/test_graphshapiq.py): neighborhoods, coalition generation, Möbius transform, efficiency tests, exact comparisons.
- [tests/shapiq/graph/test_graph_explainer.py](tests/shapiq/graph/test_graph_explainer.py): API integration, batching, overrides, budget checks.

Suggested local checks:

```bash
uv run pytest tests/shapiq/graph -q
uv run pre-commit run --all-files
```

---

## Quick Start: Reading Order for Reviewers

Use this order to review with minimal context switching.

1. Package entrypoint -> [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py)
   Confirm exposed API and import surface.
2. Core game abstraction -> [src/shapiq/graph/base.py](src/shapiq/graph/base.py)
   Validate game semantics and masking behavior.
3. Core algorithm -> [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py)
   Validate neighborhood logic, coalition pruning, Möbius computation, and efficiency handling.
4. Explainer integration -> [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py)
   Validate public API behavior and algorithm dispatch.
5. Supplementary approximation -> [src/shapiq/graph/l_shapley.py](src/shapiq/graph/l_shapley.py)
   Validate the budget-aware approximation path.
6. Integration points -> [src/shapiq/explainer/utils.py](src/shapiq/explainer/utils.py), [src/shapiq/__init__.py](src/shapiq/__init__.py), [src/shapiq/explainer/__init__.py](src/shapiq/explainer/__init__.py)
   Confirm registry and export behavior.
7. Tests -> [tests/shapiq/graph](tests/shapiq/graph)
   Prioritize algorithm correctness and exact comparisons.
8. Demo artifacts -> [src/shapiq/graph/demo](src/shapiq/graph/demo)
   Validate reproducibility assets and expected outputs.

---

## Detailed File Descriptions

### 1. [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py)

What it does:

- Exposes GraphGame, GraphSHAPIQ, and GraphExplainer as subpackage public API.

### 2. [src/shapiq/graph/base.py](src/shapiq/graph/base.py) (GraphGame)

What it does:

- Wraps a GNN + graph as a node-coalition game.

Key implementation points:

- Validates task and model.num_layers.
- Computes baseline via strategy (zeros, average, min, max, float, tensor).
- Keeps edge_index fixed while replacing node features for inactive players.
- Supports classification (class index selection) and regression.
- Supports optional normalization through empty-coalition baseline.

### 3. [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py) (GraphSHAPIQ)

What it does:

- Computes exact interactions on neighborhood-pruned coalitions.

Key implementation points:

- Builds nx.Graph from edge_index.
- Computes L-hop neighborhoods (L from model.num_layers).
- Enumerates coalition subsets up to max_subset_size.
- Computes Möbius transform with inclusion-exclusion.
- Applies efficiency routine for incomplete neighborhoods.
- Converts final Möbius values with MoebiusConverter.
- Stores last_n_model_calls.

### 4. [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py) (GraphExplainer)

What it does:

- Provides explain and explain_X interfaces for graph inputs.

Key implementation points:

- Requires torch_geometric.data.Data input.
- explain_X accepts list[Data], with optional parallelization via joblib.
- explain_function dispatches to the exact GraphSHAPIQ or LShapley path.
- Supports per-call override of index, efficiency_routine, and max_subset_size.
- Sets estimated and estimation_budget fields appropriately.

### 5. [src/shapiq/graph/l_shapley.py](src/shapiq/graph/l_shapley.py) (LShapley)

What it does:

- Approximates node-level Shapley values under a budget.

Key implementation points:

- Reuses neighborhood computation.
- Enumerates local coalitions and computes weighted marginal contributions.
- Exposes exceeded-budget signaling and call counts.

---

## Demo & Validation Artifacts

Location: [src/shapiq/graph/demo](src/shapiq/graph/demo)

The demo is organized into two main tracks:

1. **Benzene reproduction** (paper-style setup)
   - Notebook: [src/shapiq/graph/demo/benzene.ipynb](src/shapiq/graph/demo/benzene.ipynb)
   - Dataset snapshot: [src/shapiq/graph/demo/benzene_data/benzene.npz](src/shapiq/graph/demo/benzene_data/benzene.npz)
   - Trained checkpoints (GCN/GIN/GAT, 1-4 layers): [src/shapiq/graph/demo/models](src/shapiq/graph/demo/models)
   - Training curves (accuracy/loss): [src/shapiq/graph/demo/training_plots](src/shapiq/graph/demo/training_plots)
   - Result plots: [src/shapiq/graph/demo/results](src/shapiq/graph/demo/results)

2. **B-XAIC transfer study** (GraphSHAP-IQ on a new dataset)
   - Training workflow: [src/shapiq/graph/demo/b-xaic_training.ipynb](src/shapiq/graph/demo/b-xaic_training.ipynb)
   - Explainer workflow and comparison plots: [src/shapiq/graph/demo/b-xaic_explainers.ipynb](src/shapiq/graph/demo/b-xaic_explainers.ipynb)
   - Dataset file: [src/shapiq/graph/demo/b-xaic/data.csv](src/shapiq/graph/demo/b-xaic/data.csv)
   - Task folders: `B`, `P`, `PAINS`, `X`, `indole`, `rings-count`, `rings-max`
   - Per-task checkpoints: [src/shapiq/graph/demo/models/b-xaic](src/shapiq/graph/demo/models/b-xaic)
   - Per-task training plots: [src/shapiq/graph/demo/training_plots/b-xaic](src/shapiq/graph/demo/training_plots/b-xaic)
   - Explanation result plots: [src/shapiq/graph/demo/results/plots](src/shapiq/graph/demo/results/plots)

Additional notes:

- [src/shapiq/graph/demo/requirements.txt](src/shapiq/graph/demo/requirements.txt) contains notebook-specific dependencies.
- The `results/plots` folder includes both GraphSHAP-IQ and L-Shapley visual outputs for multiple architectures and graph selections.
---

## References

- Muschalik, Fumagalli, Frazzetto, Strotherm, Hermes, Sperduti, Huellermeier, Hammer.
  Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks.
  ICLR 2025. https://arxiv.org/abs/2501.16944
- Muschalik et al.
  Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles.
  AAAI 2024. https://arxiv.org/abs/2401.12069
