"""This module contains the GraphGame for GraphSHAP-IQ."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from torch_geometric.nn.models import GAT, GCN, GIN

if TYPE_CHECKING:
    from torch_geometric.data import Data

from shapiq.game import Game


class GraphGame(Game):
    """A GraphSHAP-IQ explanation game for graph networks.

    The game is based on the GraphSHAP-IQ algorithm and is used to explain the predictions of graph
    networks. GraphSHAP-IQ is used to compute Shapley interaction values for graph networks.
    """

    def __init__(
        self,
        model: GCN | GIN | GAT, 
        x_graph: Data,
        *,
        task: Literal["classification", "regression"] = "classification",
        class_index: int | None = None,
        baseline_strategy: str | None = None, 
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        """Initialize the GraphGame.

        Args:
            model: A GNN model (only GCN, GIN, or GAT) used to compute predictions.
            x_graph: The input graph as a torch_geometric Data object.
            task: Whether the model performs "classification" or "regression".
            class_index: Target class index for classification. If None, the predicted class is
                used. Must not be set when task is "regression".
            baseline_strategy: Strategy for replacing masked node features. One of "average",
                "min", "max", or "zeros". If None, defaults to zeros with a warning.
            normalize: Whether to normalize the game by the empty coalition prediction.
            verbose: Whether to show progress bars during evaluation.
        """
        if task not in ("classification", "regression"):
            raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
        if task == "regression" and class_index is not None:
            raise ValueError("class_index cannot be set for regression tasks.")

        if not isinstance(model, (GCN, GIN, GAT)):
            raise ValueError(f"Model must be GCN, GIN, or GAT, got {type(model).__name__!r}.")

        self.task = task
        self.model = model
        self.model.eval()
        self.x_graph = x_graph.clone()

        if baseline_strategy is None:
            warnings.warn(
                "Baseline is not provided, baseline will be initialized as zero...", stacklevel=2
            )
            self.baseline = torch.zeros(x_graph.num_node_features, dtype=torch.float32)
        else:
            self.baseline = self.calculate_baseline(baseline_strategy)

        if task == "classification":
            if class_index is None:
                model_output = self.model(
                    x=self.x_graph.x, edge_index=self.x_graph.edge_index, batch=self.x_graph.batch
                )
                self.y_index: int | None = int(np.argmax(model_output.detach().numpy(), axis=1)[0])
            else:
                self.y_index = int(class_index)
        else:
            self.y_index = None

        if normalize:
            normalization_value = float(self.value_function(np.zeros(len(x_graph.x))))
            super().__init__(
                n_players=len(x_graph.x),
                normalize=normalize,
                normalization_value=normalization_value,
                verbose=verbose,
            )
        else:
            super().__init__(n_players=len(x_graph.x), normalize=normalize)
        self._grand_coalition_set = set(range(self.n_players))

    def calculate_baseline(self, strategy: str) -> torch.Tensor:
        """Returns a tensor for replacing node features depending on the chosen strategy."""

        # No deep copy here, since the x_graph is not modified
        x = self.x_graph.x
        match strategy:
            case "average":
                return x.mean(dim=0)
            case "min":
                return torch.amin(x, dim=0)
            case "max":
                return torch.amax(x, dim=0)
            case "zeros":

                # Device is needed for the zeros tensor -> possible that the device is not the same as the model
                return torch.zeros(self.x_graph.num_node_features, dtype=torch.float32,
                                   device=x.device)
            case _:
                warnings.warn(
                    "Unknown baseline strategy, baseline will be initialized as zero...",
                    stacklevel=2,
                )
                return torch.zeros(self.x_graph.num_node_features, dtype=torch.float32,
                                   device=x.device)
                
    #Q: Shoudl we use both strategies for masking?
    def mask_input(self, coalition: np.ndarray) -> Data:
        """Mask inactive node features with the baseline.

        Args:
            coalition: A binary numpy array where 1 = active node, 0 = inactive.

        Returns:
            A cloned graph with inactive nodes replaced by the baseline features.
        """

        # Convert coalition to boolean tensor on the same device as the model
        coalition_tensor = torch.tensor(coalition, dtype=torch.bool, device=self.x_graph.x.device)
        x_masked = self.x_graph.clone()

        # Reshape the baseline to match the number of features in the graph
        baseline_reshaped = self.baseline.reshape(1, -1)
        x_masked.x[~coalition_tensor] = baseline_reshaped

        return x_masked

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the GNN for each coalition by masking inactive node features.

        Args:
            coalitions: Binary matrix of shape (n_coalitions, n_nodes). A 1D array of shape
                (n_nodes,) is also accepted and reshaped automatically.

        Returns:
            Array of shape (n_coalitions,) containing one model prediction per coalition.
        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        coalition_values = []

        for coalition in coalitions:
            masked_graph = self.mask_input(coalition)

            with torch.no_grad():
                model_output = self.model(
                    x=masked_graph.x,
                    edge_index=masked_graph.edge_index,
                    batch=getattr(masked_graph, "batch", None),
                )

            if self.task == "classification":
                # Output shape: (1, num_classes).
                # Select the score/logit of the target class.
                coalition_value = model_output[0, self.y_index]
            else:
                # Output is a single value.
                coalition_value = model_output.squeeze()

            coalition_values.append(float(coalition_value))

        return np.asarray(coalition_values)
