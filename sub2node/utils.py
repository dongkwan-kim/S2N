import itertools
import random
import time
from collections import namedtuple
from functools import reduce
from pprint import pprint
from typing import Dict, Any, List, Tuple, Optional, Callable, Union

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_sparse
from termcolor import cprint
from torch import Tensor
from torch_geometric.utils import dropout_adj, to_dense_batch, softmax, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor
from tqdm import tqdm


class EternalIter:

    def __init__(self, iterator):
        self.iterator = iterator

    def __iter__(self):
        return iter(self.iterator)


class ItertoolsIter:

    def __init__(self, name, *iterators, use_enumerator=False):
        self.name = name
        assert name in ["product"]
        self.iterators = iterators
        self.use_enumerator = use_enumerator

    def __iter__(self):
        if not self.use_enumerator:
            its = [it for it in self.iterators]
        else:
            its = [enumerate(it) for it in self.iterators]
        if self.name == "product":
            return itertools.product(*its)
        else:
            raise ValueError(f"Wrong name: {self.name}")

    def __len__(self):
        length = 1
        for it in self.iterators:
            length *= len(it)
        return length


__MAGIC__ = "This is magic, please trust the author."


def _try_do(do: Callable,
            obj, name_list: List[Any],
            default: Optional[Any] = __MAGIC__, iter_all=True,
            as_dict=True) -> Union[Dict[Any, Any], List, Any]:
    _e = None
    ret_list = list()
    real_name_list = []
    for name in name_list:
        try:
            ret_list.append(do(obj, name))
            real_name_list.append(name)
            if not iter_all:
                break
        except Exception as e:
            ret_list.append(default)

    if as_dict:
        return {name: ret for (name, ret) in zip(name_list, ret_list)
                if ret != __MAGIC__}
    else:
        if default != __MAGIC__:  # default is given
            return ret_list
        else:
            return [ret for ret in ret_list if ret != default]


def try_getattr(obj, name_list: List[str],
                default: Optional[Any] = __MAGIC__, iter_all=True,
                as_dict=True) -> Union[Dict[str, Any], List, Any]:
    return _try_do(do=getattr, obj=obj, name_list=name_list, default=default,
                   iter_all=iter_all, as_dict=as_dict)


def try_get_from_dict(o, name_list: List[Any],
                      default: Optional[Any] = __MAGIC__, iter_all=True,
                      as_dict=True) -> Union[Dict[Any, Any], List, Any]:
    return _try_do(do=(lambda _o, _n: _o.get(_n)),
                   obj=o, name_list=name_list, default=default,
                   iter_all=iter_all, as_dict=as_dict)


def get_log_func(func=print, **kwargs):
    def _func(*_args, **_kwargs):
        kwargs.update(**_kwargs)
        func(*_args, **kwargs)

    return _func


def ld_to_dl(list_of_dict: List[Dict[Any, Any]]) -> Dict[Any, List]:
    # https://stackoverflow.com/a/33046935
    return {k: [_dict[k] for _dict in list_of_dict] for k in list_of_dict[0]}


def iter_transform(iterator, transform: Callable = None):
    for it in iterator:
        yield it if transform is None else transform(it)


def iter_ft(iterator, transform: Callable = None, condition: Callable = None):
    iterator_f = iterator if condition is None else filter(condition, iterator)
    return iter_transform(iterator_f, transform)


def merge_dict_by_keys(first_dict: dict, second_dict: dict, keys: list):
    for k in keys:
        if k in second_dict:
            first_dict[k] = second_dict[k]
    return first_dict


def merge_dict(dict_list: List[Dict]) -> Dict:
    # https://stackoverflow.com/a/16048368
    return reduce(lambda a, b: dict(a, **b), dict_list)


def merge_dict_by_reducing_values(dict_list: List[Dict], reduce_values: Callable = sum):
    def _reduce(the_dict, a_dict):
        for k, v in a_dict.items():
            if k in the_dict:
                v = reduce_values([v, the_dict[k]])
            the_dict[k] = v
        return the_dict

    return reduce(_reduce, dict_list, {})


def startswith_any(string: str, prefix_list, *args, **kwargs) -> bool:
    for prefix in prefix_list:
        if string.startswith(prefix, *args, **kwargs):
            return True
    else:
        return False


def exist_attr(obj, name):
    return hasattr(obj, name) and (getattr(obj, name) is not None)


def del_attrs(o, keys: List[str]):
    for k in keys:
        delattr(o, k)


def rename_attr(obj, old_name, new_name):
    # https://stackoverflow.com/a/25310860
    obj.__dict__[new_name] = obj.__dict__.pop(old_name)


def func_compose(*funcs):
    # compose(f1, f2, f3)(x) == f3(f2(f1(x)))
    # https://stackoverflow.com/a/16739663
    funcs = [_f for _f in funcs if _f is not None]
    return lambda x: reduce(lambda acc, f: f(acc), funcs, x)


def debug_with_exit(func):  # Decorator
    def wrapped(*args, **kwargs):
        print()
        cprint("===== DEBUG ON {}=====".format(func.__name__), "red", "on_yellow")
        func(*args, **kwargs)
        cprint("=====   END  =====", "red", "on_yellow")
        exit()

    return wrapped


def print_time(method):  # Decorator
    """From https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d"""

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        cprint('%r  %2.2f s' % (method.__name__, (te - ts)), "red")
        return result

    return timed


def replace_all(s: str, old_to_new_values: Dict[str, str]):
    for old_value, new_value in old_to_new_values.items():
        s = s.replace(old_value, new_value)
    return s


def repr_kvs(sep="_", **kwargs):
    kvs = []
    for k, v in kwargs.items():
        kvs.append(f"{k}={v}")
    return sep.join(kvs)


# PyTorch/PyTorch Geometric related methods


def make_deterministic_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def act(tensor, activation_name, **kwargs):
    if activation_name is None:
        return tensor
    elif activation_name == "relu":
        return F.relu(tensor, **kwargs)
    elif activation_name == "elu":
        return F.elu(tensor, **kwargs)
    elif activation_name == "leaky_relu":
        return F.leaky_relu(tensor, **kwargs)
    elif activation_name == "sigmoid":
        return torch.sigmoid(tensor)
    elif activation_name == "tanh":
        return torch.tanh(tensor)
    else:
        raise ValueError(f"Wrong activation name: {activation_name}")


def get_extra_repr(model, important_args):
    return "\n".join(["{}={},".format(a, getattr(model, a)) for a in important_args
                      if a in model.__dict__])


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def spspmm_quad(m_index, m_value, a_index, a_value, k, n, coalesced=False) -> Tuple[Tensor, Tensor]:
    """
    :param m_index: sparse matrix indices of (k, n) shape
    :param m_value: values of M
    :param a_index: sparse matrix indices of (n, n) shape
    :param a_value: values of A
    :param k: the first dimension of M
    :param n: the second dimension of M
    :param coalesced: If set to :obj:`True`, will coalesce both input sparse matrices.
    :return: indices and values of M * A * M^T, a sparse matrix of (k, k) shape
    """
    # (k, n) --> (n, k)
    m_t, m_t_values = torch_sparse.transpose(m_index, m_value, k, n)
    # (k, n) * (n * n) --> (k, n)
    ma_index, ma_values = torch_sparse.spspmm(m_index, m_value, a_index, a_value, k, n, n, coalesced=coalesced)
    # (k, n) * (n, k) --> (k, k)
    return torch_sparse.spspmm(ma_index, ma_values, m_t, m_t_values, k, n, k, coalesced=coalesced)


def to_symmetric_matrix(matrix: torch.Tensor, direction="upper2lower"):
    diag = torch.diag(torch.diagonal(matrix))
    if direction == "upper2lower":
        triu = torch.triu(matrix, diagonal=1)
        return diag + triu + triu.t()
    if direction == "lower2upper":
        tril = torch.tril(matrix, diagonal=1)
        return diag + tril + tril.t()
    else:
        raise NotImplementedError


def torch_setdiff1d(tensor_1: Tensor, tensor_2: Tensor):
    dtype, device = tensor_1.dtype, tensor_1.device
    o = np.setdiff1d(tensor_1.numpy(), tensor_2.numpy())
    return torch.tensor(o, dtype=dtype, device=device)


def to_index_chunks_by_values(tensor_1d: Tensor, verbose=True) -> Dict[Any, Tensor]:
    tensor_1d = tensor_1d.flatten()
    index_chunks_dict = dict()
    # todo: there can be more efficient way, but it might not be necessary.
    v_generator = tqdm(torch.unique(tensor_1d), desc="index_chunks") if verbose else torch.unique(tensor_1d)
    for v in v_generator:
        v = v.item()
        index_chunk = torch.nonzero(tensor_1d == v).flatten()
        index_chunks_dict[v] = index_chunk
    return index_chunks_dict


def softmax_half(src: Tensor, index: Tensor, num_nodes: Optional[int] = None) -> Tensor:
    r"""softmax that supports torch.half tensors.
        See torch_geometric.utils.softmax for more details."""
    is_half = (src.dtype == torch.half)
    src = src.float() if is_half else src
    smx = softmax(src, index, num_nodes=num_nodes)
    return smx.half() if is_half else smx


def to_multiple_dense_batches(
        x_list: List[Tensor],
        batch=None, fill_value=0, max_num_nodes=None
) -> Tuple[List[Tensor], Tensor]:
    cat_x = torch.cat(x_list, dim=-1)
    cat_out, mask = to_dense_batch(cat_x, batch, fill_value, max_num_nodes)
    # [B, N, L*F] -> [B, N, F] * L
    return torch.chunk(cat_out, len(x_list), dim=-1), mask


def subgraph_and_edge_mask(subset, edge_index, edge_attr=None, relabel_nodes=False,
                           num_nodes=None):
    """Same as the pyg.utils.subgraph except it returns edge_mask too."""
    device = edge_index.device

    if isinstance(subset, list) or isinstance(subset, tuple):
        subset = torch.tensor(subset, dtype=torch.long)

    if subset.dtype == torch.bool or subset.dtype == torch.uint8:
        n_mask = subset

        if relabel_nodes:
            n_idx = torch.zeros(n_mask.size(0), dtype=torch.long,
                                device=device)
            n_idx[subset] = torch.arange(subset.sum().item(), device=device)
    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        n_mask = torch.zeros(num_nodes, dtype=torch.bool)
        n_mask[subset] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[subset] = torch.arange(subset.size(0), device=device)

    mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask] if edge_attr is not None else None

    if relabel_nodes:
        edge_index = n_idx[edge_index]

    return edge_index, edge_attr, mask


def idx_to_mask(idx_dict: Dict[Any, Tensor], num_nodes: int):
    mask_dict = dict()
    for k, idx in idx_dict.items():
        # idx: LongTensor
        mask = torch.zeros((num_nodes,), dtype=torch.bool)
        mask[idx] = 1
        mask_dict[k] = mask
    return mask_dict


def mean_std(values: Union[np.ndarray, torch.Tensor, List]) -> Tuple[float, float]:
    if isinstance(values, torch.Tensor):
        values = values.float().tolist()
    mean = np.mean(values)
    std = np.std(values)
    return float(mean), float(std)


def dropout_adj_st(edge_index: Union[Tensor, SparseTensor],
                   edge_attr=None, p=0.5, force_undirected=False,
                   num_nodes=None, training=True):
    use_sparse_tensor = isinstance(edge_index, SparseTensor)
    if use_sparse_tensor:
        row, col, edge_attr = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
        N = maybe_num_nodes(edge_index, num_nodes)

    edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=p,
                                        force_undirected=force_undirected,
                                        num_nodes=num_nodes, training=training)
    if use_sparse_tensor:
        adj = SparseTensor.from_edge_index(edge_index, edge_attr, sparse_sizes=(N, N))
        return adj, None
    else:
        return edge_index, edge_attr


def filter_living_edge_index(edge_index, num_nodes, min_index=None):
    mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
    if min_index is not None:
        mask = mask & (edge_index[0] >= min_index) & (edge_index[1] >= min_index)
    return edge_index[:, mask]


def torch_choice(tensor_1d: Tensor, sample_size_or_ratio: int or float):
    tensor_size = tensor_1d.size(0)
    if isinstance(sample_size_or_ratio, float):
        sample_size_or_ratio = int(sample_size_or_ratio * tensor_size)
    perm = torch.randperm(tensor_size)
    return tensor_1d[perm[:sample_size_or_ratio]]


def multi_label_homophily(edge_index, y, batch=None, method="edge") -> float:
    # y: [N, C]
    if isinstance(edge_index, SparseTensor):
        col, row, _ = edge_index.coo()
    else:
        row, col = edge_index

    y = y.bool()
    y_row, y_col = y[row], y[col]
    y_inter = y_row & y_col
    y_union = y_row | y_col

    if method == "edge":
        # ml_eh = 1 / |E| \sum_{(u, v) \in E} ( | C_u ∩ C_v | / | C_u ∪ C_v | )
        out = y_inter.sum(dim=-1) / y_union.sum(dim=-1)
        if batch is None:
            return float(out.mean())
        else:
            return scatter_mean(out, batch[col], dim=0)

    else:
        # ml_nh = 1 / |V| \sum_u \sum_{v \in N(u)} ( | C_u ∩ C_v | / | C_u ∪ C_v | ) / |N(u)|
        out = y_inter.sum(dim=-1) / y_union.sum(dim=-1)
        out = scatter_mean(out, col, 0, dim_size=y.size(0))
        if batch is None:
            return float(out.mean())
        else:
            return scatter_mean(out, batch, dim=0)


# networkx


def to_directed(edge_index, edge_attr=None):
    if edge_attr is not None:
        raise NotImplementedError
    N = edge_index.max().item() + 1
    row, col = torch.sort(edge_index.t()).values.t()
    sorted_idx = torch.unique(row * N + col)
    row, col = sorted_idx // N, sorted_idx % N
    return torch.stack([row, col], dim=0).long()


def convert_node_labels_to_integers_customized_ordering(
        G, first_label=0, ordering="default", label_attribute=None
):
    if ordering == "keep":
        mapping = dict(zip(G.nodes(), [int(v) for v in G.nodes()]))
        H = nx.relabel_nodes(G, mapping)
        if label_attribute is not None:
            nx.set_node_attributes(H, {v: k for k, v in mapping.items()}, label_attribute)
        return H
    else:
        return nx.convert_node_labels_to_integers(G, first_label, ordering, label_attribute)


def from_networkx_customized_ordering(G, ordering="default"):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.
    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    G = convert_node_labels_to_integers_customized_ordering(G, ordering=ordering)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    return data


def unbatch(src: Tensor, batch: Tensor, dim: int = 0) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)

    :rtype: :class:`List[Tensor]`

    NOTE: Copied from https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


if __name__ == '__main__':

    METHOD = "filter_living_edge_index"

    from pytorch_lightning import seed_everything

    seed_everything(42)

    if METHOD == "iter_ft":
        pprint(list(iter_ft(
            dict(enumerate(["a", "b", "c", "d", "e"])).items(),
            transform=lambda kv: kv[1] + "/p",
            condition=lambda kv: kv[0] % 2 == 0,
        )))

    elif METHOD == "multi_label_node_homophily":

        _ei = torch.tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]]).long()

        _mlnh = multi_label_homophily(
            edge_index=_ei,
            y=torch.Tensor([[1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 1, 0]]).float(),
        )
        print(_mlnh)

        _mlnh = multi_label_homophily(
            edge_index=_ei,
            y=torch.Tensor([[1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0],
                            [1, 0, 0]]).float(),
        )
        print(_mlnh)

        _mlnh = multi_label_homophily(
            edge_index=_ei,
            y=torch.Tensor([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0]]).float(),
        )
        print(_mlnh)

        _mlnh = multi_label_homophily(
            edge_index=_ei,
            y=torch.Tensor([[1, 1, 1],
                            [1, 1, 0],
                            [1, 0, 0],
                            [0, 1, 0]]).float(),
        )
        print(_mlnh)

        _mlnh = multi_label_homophily(
            edge_index=_ei,
            y=torch.Tensor([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]).float(),
        )
        print(_mlnh)

        _mlnh = multi_label_homophily(
            edge_index=_ei,
            y=torch.Tensor([[0, 0, 0],
                            [0, 0, 0],
                            [1, 1, 1],
                            [1, 1, 1]]).float(),
        )
        print(_mlnh)

    elif METHOD == "ld_to_dl":
        print(ld_to_dl([{}]))
        print(ld_to_dl([{"x": 1}, {"x": 2}]))

    elif METHOD == "try_getattr":
        Point = namedtuple('Point', ['x', 'y'])
        _p = Point(x=11, y=22)
        pprint(try_getattr(_p, ["x", "z", "y"]))
        pprint(try_getattr(_p, ["x", "z", "y"], default=None))
        pprint(try_getattr(_p, ["x", "z", "y"], as_dict=False))
        pprint(try_getattr(_p, ["x", "z", "y"], as_dict=False, default=None))

    elif METHOD == "try_get_from_dict":
        pprint(try_get_from_dict({"x": 1, "y": 2}, ["x", "z", "y"]))

    elif METHOD == "get_log_func":
        print_red = get_log_func(cprint, color="red")
        print_red("this is red print")

    elif METHOD == "to_index_chunks_by_values":
        _tensor_1d = torch.Tensor([24, 20, 21, 21, 20, 23, 24])
        print(to_index_chunks_by_values(_tensor_1d))

    elif METHOD == "try_get_from_dict":
        pprint(try_get_from_dict({1: 10, 2: 20, 3: 30},
                                 name_list=[3, 2, 123]))

    elif METHOD == "merge_dict_by_reducing_values":
        pprint(merge_dict_by_reducing_values(
            [{"a": 1, "c": 3},
             {"a": 10, "b": 20, "c": 30},
             {"a": 100, "b": 200}],
            reduce_values=sum))

    elif METHOD == "make_symmetric":
        m = torch.Tensor([[1, 2, 3],
                          [-1, 4, 5],
                          [-1, -1, 6]])
        print(to_symmetric_matrix(m))

    elif METHOD == "dropout_adj_st":
        _edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                    [1, 0, 2, 1, 3, 2]])
        _edge_attr = torch.Tensor([1, 2, 3, 4, 5, 6])

        _row, _col = _edge_index
        _adj = SparseTensor(row=_row, col=_col, value=_edge_attr, sparse_sizes=(4, 4))

        print(_adj)
        print(dropout_adj_st(_adj, p=0.9)[0])

    elif METHOD == "filter_living_edge_index":
        _edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                                    [1, 0, 2, 1, 3, 2]])
        print(filter_living_edge_index(_edge_index, num_nodes=3))
        print(filter_living_edge_index(_edge_index, num_nodes=4))
        print(filter_living_edge_index(_edge_index, num_nodes=5))

    elif METHOD == "ItertoolsIter":
        _pi = ItertoolsIter("product", [1, 2, 3], [1, 2, 3])
        for _emt_pi in _pi:
            print(_emt_pi)
        print("len", len(_pi))

    elif METHOD == "repr_kvs":
        print(repr_kvs(mmt="string", en=123, ucp=True, sep=", "))

    else:
        raise ValueError("Wrong method: {}".format(METHOD))
