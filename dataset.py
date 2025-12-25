"""Dataset utilities for constructing torch-geometric hetero datasets."""
from __future__ import annotations

import argparse
import os
import pickle
import time
from typing import Iterable, List, Optional, Tuple

import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build HeteroDataset from preprocessed files.")
    parser.add_argument("-r", "--root", default="./data", help="Root directory of the dataset")
    parser.add_argument("--dataset_name", required=True, help="Dataset file name")
    parser.add_argument("--pos_encoding_dim", type=int, default=0, help="Dim of positional encoding")
    parser.add_argument("--bus_region_table", default=None, help="Optional path to bus region tensor")
    parser.add_argument("--bus_id_region_map", default=None, help="Optional path to bus id->region map tensor")
    return parser


def load_dataset_file(path: str):
    try:
        return torch.load(path, weights_only=False)
    except (RuntimeError, pickle.UnpicklingError, EOFError):
        with open(path, "rb") as f:
            return pickle.load(f)


def as_tensor(value, dtype=None):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype) if dtype is not None else value
    return torch.as_tensor(value, dtype=dtype)


def split_matrix_attr(matrix_attr: Optional[torch.Tensor]):
    if matrix_attr is None:
        return None, None, None, None
    if matrix_attr.ndim != 2 or matrix_attr.shape[1] < 2:
        raise ValueError("matrix_attr must be 2D with at least 2 columns for [G, Bpp].")
    g_bpp = matrix_attr[:, :2]
    J = matrix_attr[:, 2:6] if matrix_attr.shape[1] >= 6 else None
    Bp = matrix_attr[:, 6] if matrix_attr.shape[1] >= 7 else None
    g = g_bpp[:, 0]
    bpp = g_bpp[:, 1]
    return g, bpp, J, Bp


class HeteroDataset(pyg.data.InMemoryDataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        transform=None,
        pre_transform=None,
        pos_encoding_dim: int = 0,
        bus_region_table=None,
        bus_id_region_map: torch.Tensor = None,
    ):
        self.dataset_name = dataset_name
        self.pos_enc_dim = pos_encoding_dim
        self.bus_region_table = bus_region_table
        self.lookup = self._build_lookup_table(bus_id_region_map) if bus_id_region_map is not None else None
        self.lpe_cache = {}
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return [self.dataset_name]

    def process(self) -> None:
        start_time = time.time()
        path = os.path.join(self.root, self.dataset_name)
        dataset = load_dataset_file(path)
        print(f"Successfully loaded the dataset from: {path}")

        if isinstance(dataset, list):
            records = dataset
        else:
            records = self._convert_dict_dataset(dataset)

        g_dataset = []
        for record in tqdm(records):
            g_dataset.append(self._build_single_data(record))

        data, slices = self.collate(g_dataset)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Single-file] Saved {len(g_dataset)} samples to {self.processed_paths[0]}")
        print(f"Finished processing {len(g_dataset)} samples: {round(time.time() - start_time, 2)} sec.")

    def _build_single_data(self, record: dict) -> HeteroData:
        x = as_tensor(record.get("x"), dtype=torch.float64)
        y = as_tensor(record.get("y"), dtype=torch.float64)

        matrix = record.get("matrix_attr") or {}
        sp_incidence = as_tensor(matrix.get("incidence"), dtype=torch.long)
        matrix_attr = as_tensor(matrix.get("attr"), dtype=torch.float64)
        g, bpp, jacobian, bp = split_matrix_attr(matrix_attr)

        branch = record.get("branch_attr") or {}
        br_incidence = as_tensor(branch.get("incidence"), dtype=torch.long)
        br_attr = as_tensor(branch.get("attr"), dtype=torch.float64)
        br_param = br_attr[:, :5] if br_attr is not None else None
        br_flow = br_attr[:, 5:9] if br_attr is not None and br_attr.shape[1] >= 9 else None

        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        to_f64 = lambda t: t.to(device=device, dtype=torch.float64)
        to_i64 = lambda t: t.to(device=device, dtype=torch.long)

        g_data = HeteroData()
        num_nodes = y.shape[0]
        g_data["bus"].num_nodes = int(num_nodes)
        g_data["bus"].y = to_f64(y)

        features = [to_f64(x[:, :4])]
        if self.pos_enc_dim > 0:
            features.append(
                self.add_pos_encoding(
                    g_data,
                    edge_index_matrix=to_i64(sp_incidence),
                    edge_index_branch=to_i64(br_incidence) if br_incidence is not None else None,
                    pos_enc_dim=self.pos_enc_dim,
                )
            )
        features.append(self._bus_region_feature(x, num_nodes, device))
        features.append(x[:, -1].unsqueeze(-1))
        g_data["bus"].x = torch.cat(features, dim=-1)

        matrix_rel = ("bus", "matrix", "bus")
        g_data[matrix_rel].edge_index = to_i64(sp_incidence)
        if g is not None:
            g_data[matrix_rel].g = to_f64(g)
        if bpp is not None:
            g_data[matrix_rel].bpp = to_f64(bpp)
        if bp is not None:
            g_data[matrix_rel].bp = to_f64(bp)
        if jacobian is not None:
            g_data[matrix_rel].jacobian = to_f64(jacobian)

        if br_incidence is not None:
            branch_rel = ("bus", "connect", "bus")
            g_data[branch_rel].edge_index = to_i64(br_incidence)
            if br_param is not None:
                g_data[branch_rel].param = to_f64(br_param)
            if br_flow is not None:
                g_data[branch_rel].flow = to_f64(br_flow)

        g_data = self._largest_connected_component(g_data, ref_edge_type=matrix_rel)

        if self.pre_transform is not None:
            g_data = self.pre_transform(g_data)

        return g_data

    def _bus_region_feature(self, x: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        if self.bus_region_table is not None:
            region = torch.as_tensor(self.bus_region_table, device=device)
            if region.dim() == 1:
                region = region.unsqueeze(-1)
            if region.shape[0] != num_nodes:
                raise ValueError(f"bus_region_table rows {region.shape[0]} != num_nodes {num_nodes}")
            return region.to(dtype=torch.float64)
        if self.lookup is not None:
            bus_region = self.lookup[x[:, -2].type(torch.long)].unsqueeze(-1)
            return bus_region.to(dtype=torch.float64)
        return x[:, -2].unsqueeze(-1).to(dtype=torch.float64)

    def add_pos_encoding(
        self,
        data: HeteroData,
        pos_enc_dim: int,
        relation: str = "matrix",
        edge_index_matrix: torch.Tensor = None,
        edge_index_branch: torch.Tensor = None,
    ) -> torch.Tensor:
        num_nodes = data["bus"].num_nodes
        device = edge_index_matrix.device if edge_index_matrix is not None else torch.device("cpu")
        if num_nodes == 1 or pos_enc_dim <= 0:
            return torch.zeros((num_nodes, pos_enc_dim), device=device, dtype=torch.float32)

        if relation == "matrix":
            edge_index = edge_index_matrix
        elif relation == "connect":
            edge_index = edge_index_branch
        else:
            raise ValueError("relation must be 'matrix' or 'connect'")
        if edge_index is None:
            return torch.zeros((num_nodes, pos_enc_dim), device=device, dtype=torch.float32)

        cache_key = (int(num_nodes), int(pos_enc_dim), relation, edge_index.detach().cpu().numpy().tobytes())
        if cache_key in self.lpe_cache:
            return self.lpe_cache[cache_key].to(device)

        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        deg = adj.sum(dim=1)
        d_inv_sqrt = torch.diag(1.0 / torch.sqrt(deg + 1e-6))
        laplacian = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype) - d_inv_sqrt @ adj @ d_inv_sqrt

        _, evecs = torch.linalg.eigh(laplacian)
        pos_enc = evecs[:, :pos_enc_dim].to(torch.float32)
        self.lpe_cache[cache_key] = pos_enc.detach().cpu()
        return pos_enc.to(device)

    def _largest_connected_component(
        self,
        data_hetero: HeteroData,
        ref_edge_type: Tuple[str, str, str] = ("bus", "matrix", "bus"),
        num_components: int = 1,
        connection: str = "weak",
        remove_selfloops: bool = True,
    ) -> HeteroData:
        node_types: List[str] = list(data_hetero.node_types)
        edge_types: List[Tuple[str, str, str]] = list(data_hetero.edge_types)
        if len(node_types) != 1:
            raise ValueError(f"Expected 1 node type, got {len(node_types)}: {node_types}")
        if ref_edge_type not in edge_types:
            raise ValueError(f"{ref_edge_type} not in edge types: {edge_types}")

        ntype = node_types[0]
        num_nodes = int(data_hetero[ntype].num_nodes)
        edge_index = data_hetero[ref_edge_type].edge_index

        graph = nx.Graph() if connection.lower() == "weak" else nx.DiGraph()
        graph.add_nodes_from(range(num_nodes))

        if edge_index.numel() > 0:
            src = edge_index[0].tolist()
            dst = edge_index[1].tolist()
            edges = [(u, v) for u, v in zip(src, dst) if (u != v or not remove_selfloops)]
            graph.add_edges_from(edges)

        comps = (
            list(nx.connected_components(graph))
            if connection.lower() == "weak"
            else list(nx.strongly_connected_components(graph))
        )
        if not comps:
            return data_hetero.subgraph({ntype: torch.empty(0, dtype=torch.long)})

        comps.sort(key=len, reverse=True)
        keep_set = set().union(*comps[: max(1, min(num_components, len(comps)))])
        if not keep_set:
            return data_hetero.subgraph({ntype: torch.empty(0, dtype=torch.long)})

        keep_nodes = torch.tensor(sorted(keep_set), dtype=torch.long)
        return data_hetero.subgraph({ntype: keep_nodes})

    @staticmethod
    def _convert_dict_dataset(dataset: dict) -> Iterable[dict]:
        for i in range(len(dataset["x"])):
            matrix_parts = [dataset["admittance"][i], dataset["j"][i]]
            if "b_prime" in dataset:
                matrix_parts.append(dataset["b_prime"][i].unsqueeze(-1))
            record = {
                "x": dataset["x"][i],
                "y": dataset["y"][i],
                "matrix_attr": {
                    "incidence": dataset["incidence"][i],
                    "attr": torch.cat(matrix_parts, dim=-1),
                },
            }
            if "br_incidence" in dataset:
                record["branch_attr"] = {
                    "incidence": dataset.get("br_incidence")[i],
                    "attr": torch.cat([dataset.get("br_param")[i], dataset.get("sij")[i]], dim=-1),
                }
            yield record

    @staticmethod
    def _build_lookup_table(mapping: torch.Tensor) -> torch.Tensor:
        max_key = mapping[:, 0].max().item() + 1
        lookup = torch.full((max_key,), -1, dtype=mapping.dtype)
        lookup[mapping[:, 0]] = mapping[:, 1]
        return lookup


def main() -> None:
    parser = arg_parser()
    args = parser.parse_args()

    bus_region_table = torch.load(args.bus_region_table) if args.bus_region_table else None
    bus_id_region_map = torch.load(args.bus_id_region_map) if args.bus_id_region_map else None

    HeteroDataset(
        root=args.root,
        dataset_name=args.dataset_name,
        pos_encoding_dim=args.pos_encoding_dim,
        bus_region_table=bus_region_table,
        bus_id_region_map=bus_id_region_map,
    )


if __name__ == "__main__":
    main()
