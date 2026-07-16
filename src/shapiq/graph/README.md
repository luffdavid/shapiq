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

GraphSHAP-IQ computes Shapley interaction values for Graph Neural Networks (GNNs) by exploiting message-passing locality.

The core motivation is the receptive field structure of a GNN:

- After $L$ layers, a node representation depends only on its $L$-hop neighborhood.
- Coalitions outside the relevant receptive fields do not need to be evaluated for node-local interactions.
- Coalition evaluation can therefore be reduced from brute force $O(2^n)$ toward locality-aware subsets induced by graph neighborhoods.

In shapiq, this is the graph counterpart to TreeSHAP-IQ: a structure-aware exact method when all neighborhood subsets are evaluated, and a controlled approximation when coalition sizes are truncated via `max_subset_size`.

---

## High-Level Project Overview

GraphSHAP-IQ in this repository is implemented as a graph subpackage with four main pieces:

1. GraphGame: wraps a PyTorch Geometric graph/model as a cooperative game over nodes.
2. GraphSHAPIQ: builds locality-pruned coalitions, computes Möbius coefficients, optionally applies an efficiency correction, and converts them to interaction indices.
3. GraphExplainer: user-facing integration with shapiq's `Explainer` framework.
4. `cext`: C++ acceleration for the Möbius transform used by `GraphSHAPIQ.explain(..., use_cpp=True)`.

The public Python API exported by `shapiq.graph` currently consists of `GraphGame`, `GraphExplainer`, and `GraphSHAPIQ`.

---

## Core Concepts And Terminology

- **Player**: one graph node.
- **Coalition**: a subset of nodes kept active while all other node features are replaced by a baseline.
- **Baseline**: the feature vector used for inactive nodes. It is either derived from the graph (`zeros`, `average`, `min`, `max`) or passed explicitly via `baseline_value`.
- **Neighborhood**: the sorted $L$-hop receptive field of one node, where $L = model.num_layers$.
- **Möbius coefficients**: locality-pruned intermediate coefficients computed from coalition values before conversion to the requested interaction index.
- **Efficiency routine**: an optional correction that restores the efficiency axiom when the Möbius transform is truncated because `max_subset_size` is smaller than some neighborhoods.
- **Exact vs. estimated**: `GraphSHAPIQ` is exact when no truncation occurs, and also in the boundary case `max_subset_size >= max_size_neighbors - 1` when the efficiency routine is enabled. Otherwise the result is an approximation.

---

## Subpackage Architecture & File-by-File Overview

🏗️ Current implementation in [src/shapiq/graph](src/shapiq/graph):

- [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py): public subpackage exports (`GraphGame`, `GraphExplainer`, `GraphSHAPIQ`).
- [src/shapiq/graph/base.py](src/shapiq/graph/base.py): `GraphGame`.
- [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py): `GraphSHAPIQ` core algorithm and public Möbius-transform routines.
- [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py): `GraphExplainer` integration layer.
- [src/shapiq/graph/cext](src/shapiq/graph/cext): C++ sources for the Möbius-transform acceleration.
- [src/shapiq/graph/demo](src/shapiq/graph/demo): notebooks, datasets, checkpoints, and generated artifacts.

| File | Role |
| --- | --- |
| [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py) | Subpackage API |
| [src/shapiq/graph/base.py](src/shapiq/graph/base.py) | `GraphGame` |
| [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py) | Exact/truncated GraphSHAP-IQ routine |
| [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py) | User-facing explainer |
| [src/shapiq/graph/cext/cext.cc](src/shapiq/graph/cext/cext.cc)    | Python-to-C++ interface layer   |
| [src/shapiq/graph/cext/moebius.cc](src/shapiq/graph/cext/moebius.cc) | Core Moebius-transform algorithm |

---

## Package Structure

📂 Quick tree view of the graph subpackage:

```text
src/shapiq/graph/
├── README.md
├── __init__.py
├── base.py
├── cext/
│   ├── cext.cc
│   └── moebius.cc
├── demo/
│   ├── benzene.ipynb
│   ├── b-xaic_training.ipynb
│   ├── b-xaic_explainers.ipynb
│   ├── b-xaic/
│   ├── benzene_data/
│   ├── models/
│   ├── results/
│   ├── requirements.txt
│   └── training_plots/
├── explainer.py
└── graphshapiq.py
```

---

## Usage Notes
### Minimal Example

```python
from shapiq.graph import GraphExplainer

explainer = GraphExplainer(
    model=model,
    index="k-SII",
    baseline_strategy="zeros",
    max_order=2,
    class_index=None,
    efficiency_routine=True,
    normalize=True,
)

interaction_values = explainer.explain(
    x_graph,
    max_subset_size=None,
)
```

### Batch Explanation

```python
results = explainer.explain_X(
    [x_graph_1, x_graph_2],
    n_jobs=-1,
    verbose=True,
)
```

### Low-Level API

Use the lower-level classes directly if you need explicit access to Möbius coefficients, or if you need to force the pure-Python path with `use_cpp=False`.

```python
from shapiq.graph import GraphGame, GraphSHAPIQ

game = GraphGame(
    model=model,
    x_graph=x_graph,
    class_index=0,
    baseline_strategy="average",
    normalize=True,
)

explainer = GraphSHAPIQ(game=game)
moebius_values, interaction_values = explainer.explain(
    index="SII",
    order=2,
    max_subset_size=1,
    efficiency_routine=True,
    use_cpp=False,
)
```
---

## Testing And Validation & Tests to Inspect

Graph-specific tests are in [tests/shapiq/graph](tests/shapiq/graph):

- [tests/shapiq/graph/conftest.py](tests/shapiq/graph/conftest.py): model and graph fixtures (GCN/GIN/GAT and edge-case graph topologies).
- [tests/shapiq/graph/test_graph_game.py](tests/shapiq/graph/test_graph_game.py): masking, baseline handling, normalization, and value-function behavior.
- [tests/shapiq/graph/test_graphshapiq.py](tests/shapiq/graph/test_graphshapiq.py): neighborhoods, coalition generation, Möbius transforms, efficiency correction, exact-comparison checks, and Python/C++ parity.
- [tests/shapiq/graph/test_graph_explainer.py](tests/shapiq/graph/test_graph_explainer.py): public API behavior, batching, override handling, exactness metadata, and input validation.

Suggested local checks:

```bash
uv run pytest tests/shapiq/graph -q
uv run pre-commit run --all-files
```

---

## Quick Start: Reading Order for Reviewers

Use this order to review with minimal context switching.

1. Package entrypoint -> [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py)
   Confirm the public import surface.
2. Core game abstraction -> [src/shapiq/graph/base.py](src/shapiq/graph/base.py)
   Validate masking semantics, baseline construction, normalization, and model-output handling.
3. Core algorithm -> [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py)
   Validate neighborhood logic, coalition pruning, Möbius computation, efficiency handling, and exactness bookkeeping.
4. Explainer integration -> [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py)
   Validate public API behavior, `Data`-type requirements, batching, and metadata propagation.
5. Optional acceleration -> [src/shapiq/graph/cext](src/shapiq/graph/cext)
   Review only if you need to understand the compiled Möbius-transform path.
6. Tests -> [tests/shapiq/graph](tests/shapiq/graph)
   Prioritize exact comparisons, truncation behavior, and API contracts.
7. Demo artifacts -> [src/shapiq/graph/demo](src/shapiq/graph/demo)
   Validate notebooks, checkpoints, and generated outputs.

---

## Detailed File Descriptions

### 1. [src/shapiq/graph/__init__.py](src/shapiq/graph/__init__.py)

What it does:

- Exposes `GraphGame`, `GraphSHAPIQ`, and `GraphExplainer` as the subpackage public API.

### 2. [src/shapiq/graph/base.py](src/shapiq/graph/base.py) (`GraphGame`)

What it does:

- Wraps a GNN + graph as a node-coalition game.

Key implementation points:

- Requires `x_graph.x` and a model with an integer `num_layers` attribute.
- Clones the input graph and treats every node as one player.
- Computes the node-masking baseline from `baseline_strategy` (`zeros`, `average`, `min`, `max`) or from an explicit `baseline_value` (`float` or feature-wise `torch.Tensor`).
- Keeps `edge_index` fixed while replacing inactive node features.
- Supports scalar-output models directly and multi-output models via `class_index`.
- Forwards an existing `batch` attribute to the model if present.
- Optionally normalizes the game by subtracting the empty-coalition value through the shared `Game` base class.

### 3. [src/shapiq/graph/graphshapiq.py](src/shapiq/graph/graphshapiq.py) (`GraphSHAPIQ`)

What it does:

- Computes GraphSHAP-IQ Möbius coefficients and converts them to Shapley interaction indices.

Key implementation points:

- Builds an `nx.Graph` from `edge_index`.
- Computes sorted $L$-hop neighborhoods for every node, where $L = game.l_hop_distance$.
- Collects all coalition subsets up to `max_subset_size` from the per-node neighborhoods.
- Optionally tracks incomplete neighborhoods and applies an efficiency-gap correction when `efficiency_routine=True`.
- Exposes both `compute_moebius_transform` (Python) and `compute_moebius_transform_cpp` (compiled extension) as public routines.
- Uses `MoebiusConverter` to map final Möbius coefficients to indices such as `SV`, `SII`, `k-SII`, `STII`, `FSII`, and `FBII`.
- Tracks `last_n_model_calls`, `last_computation_exact`, `total_budget`, and `budget_estimated`.
- Returns both the final Möbius coefficients and the requested interaction values from `explain`.

### 4. [src/shapiq/graph/explainer.py](src/shapiq/graph/explainer.py) (`GraphExplainer`)

What it does:

- Provides the user-facing `Explainer` integration for graph inputs.

Key implementation points:

- Requires `torch_geometric.data.Data` inputs and raises a helpful optional-dependency error if graph dependencies are missing.
- Constructor surface: `model`, `index`, `baseline_strategy`, `baseline_value`, `max_order`, `class_index`, `efficiency_routine`, `sparsify_threshold`, and `normalize`.
- `explain` validates a single `Data` object and delegates to `GraphSHAPIQ`.
- `explain_X` accepts a list of `Data` objects, supports `joblib` parallelization via `n_jobs`, and optionally displays a progress bar.
- Per-call overrides are currently limited to `index`, `efficiency_routine`, and `max_subset_size`.
- Sets `InteractionValues.estimation_budget` from `GraphSHAPIQ.last_n_model_calls` and `InteractionValues.estimated` from `GraphSHAPIQ.last_computation_exact`.
- Sparsifies near-zero outputs using `sparsify_threshold`.

### 5. [src/shapiq/graph/cext](src/shapiq/graph/cext)

What it does:

- Contains the C++ sources backing `shapiq.graph.cext`, which accelerates the Möbius-transform hot loop.

Key implementation points:

- `GraphSHAPIQ.explain(..., use_cpp=True)` uses this path.
- The Python API can still be exercised explicitly with `use_cpp=False` on `GraphSHAPIQ`.
- `GraphExplainer` goes through `GraphSHAPIQ` and therefore uses its default compiled path.
- In the current public API, `GraphExplainer` does not expose a `use_cpp` switch, so the low-level `GraphGame` + `GraphSHAPIQ` route is the way to run without the compiled extension.

---

## Demo & Validation Artifacts

Location: [src/shapiq/graph/demo](src/shapiq/graph/demo)

The demo is organized into two main tracks:

1. **Benzene reproduction** (paper-style setup)
   - Notebook: [src/shapiq/graph/demo/benzene.ipynb](src/shapiq/graph/demo/benzene.ipynb)
   - Dataset snapshot: [src/shapiq/graph/demo/benzene_data/benzene.npz](src/shapiq/graph/demo/benzene_data/benzene.npz)
   - Trained checkpoints (GCN/GIN/GAT, 1-4 layers): [src/shapiq/graph/demo/models](src/shapiq/graph/demo/models)
   - Training curves: [src/shapiq/graph/demo/training_plots](src/shapiq/graph/demo/training_plots)
   - Complexity tables and plots: [src/shapiq/graph/demo/results](src/shapiq/graph/demo/results)

2. **B-XAIC transfer study**
   - Training workflow: [src/shapiq/graph/demo/b-xaic_training.ipynb](src/shapiq/graph/demo/b-xaic_training.ipynb)
   - Explainer workflow and plots: [src/shapiq/graph/demo/b-xaic_explainers.ipynb](src/shapiq/graph/demo/b-xaic_explainers.ipynb)
   - Dataset file: [src/shapiq/graph/demo/b-xaic/data.csv](src/shapiq/graph/demo/b-xaic/data.csv)
   - Per-task checkpoints: [src/shapiq/graph/demo/models/b-xaic](src/shapiq/graph/demo/models/b-xaic)
   - Per-task training plots: [src/shapiq/graph/demo/training_plots/b-xaic](src/shapiq/graph/demo/training_plots/b-xaic)
   - Explanation result plots: [src/shapiq/graph/demo/results/plots](src/shapiq/graph/demo/results/plots)

Additional notes:

- [src/shapiq/graph/demo/requirements.txt](src/shapiq/graph/demo/requirements.txt) contains notebook-specific dependencies.
- The `results/plots` folder contains both benzene plots and B-XAIC task-specific subdirectories.

---

## References

- Muschalik, Fumagalli, Frazzetto, Strotherm, Hermes, Sperduti, Huellermeier, Hammer.
  Exact Computation of Any-Order Shapley Interactions for Graph Neural Networks.
  ICLR 2025. https://arxiv.org/abs/2501.16944
- Muschalik et al.
  Beyond TreeSHAP: Efficient Computation of Any-Order Shapley Interactions for Tree Ensembles.
  AAAI 2024. https://arxiv.org/abs/2401.12069
