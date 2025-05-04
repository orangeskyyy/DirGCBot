import torch
from GCL.augmentors.augmentor import Graph, Augmentor
from GCL.augmentors.functional import drop_feature

class CustomAugmentor(Augmentor):
    def __init__(self, pf: float, pe: float, pa: float, device: torch.device):
        super(CustomAugmentor, self).__init__()
        self.pf = pf  # Feature masking probability
        self.pe = pe  # Edge removing probability
        self.pa = pa  # Edge adding probability
        self.device = device  # Device to move tensors to

    def augment(self, g: Graph) -> Graph:
        x, edge_index, edge_weights = g.unfold()

        # Move tensors to the same device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_weights = edge_weights.to(self.device)

        # Feature Masking
        x = drop_feature(x, self.pf)

        # Edge operations
        edge_index, edge_weights = self.modify_edges(edge_index, edge_weights, x.size(0))

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)

    def modify_edges(self, edge_index, edge_weights, num_nodes):
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=self.device)
        adj[edge_index[0], edge_index[1]] = True

        # Identify bidirectional edges
        bidirectional = adj & adj.t()

        new_edges = []
        new_weights = []

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()

            if bidirectional[src, dst]:
                # If bidirectional, keep the edge
                new_edges.append((src, dst))
                new_weights.append(edge_weights[i])
            else:
                # If not bidirectional, apply edge removing with probability pe
                if torch.rand(1).item() > self.pe:
                    new_edges.append((src, dst))
                    new_weights.append(edge_weights[i])

        # Convert new edges back to tensor format
        if new_edges:
            edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t()
            edge_weights = torch.tensor(new_weights, dtype=edge_weights.dtype, device=self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_weights = torch.empty((0,), dtype=edge_weights.dtype, device=self.device)

        # Apply edge adding with probability pa for nodes with common neighbors
        for node in range(num_nodes):
            neighbors = (adj[node] & adj.t()[node]).nonzero(as_tuple=False).view(-1)
            for n in neighbors:
                if torch.rand(1).item() < self.pa:
                    new_edges.append((node, n.item()))
                    new_weights.append(torch.tensor(1.0, dtype=edge_weights.dtype, device=self.device))  # Assuming a default weight of 1.0

        if new_edges:
            edge_index = torch.tensor(new_edges, dtype=torch.long, device=self.device).t()
            edge_weights = torch.tensor(new_weights, dtype=edge_weights.dtype, device=self.device)

        return edge_index, edge_weights