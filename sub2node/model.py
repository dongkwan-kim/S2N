from typing import Dict, Union, Any, List

import torch
import torch.nn as nn
from omegaconf import ListConfig
from pytorch_lightning import (LightningModule, seed_everything)
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch_geometric.data import Data

from data import SubgraphDataModule
from evaluator import Evaluator
from model_linkx import InductiveLINKX
from model_utils import GraphEncoder, VersatileEmbedding, MLP, DeepSets, Readout, GraphEncoderSequential, MyIdentity
from run_utils import get_logger
from utils import try_getattr, ld_to_dl, try_get_from_dict

log = get_logger(__name__)


class GraphNeuralModel(LightningModule):

    @property
    def h(self):
        return self.hparams

    @property
    def dh(self):
        return self.given_datamodule.hparams

    def __init__(self,
                 encoder_layer_name: Union[str, List[str]],
                 num_layers: Union[int, List[int]],
                 hidden_channels: int,
                 activation: str,
                 learning_rate: float,
                 weight_decay: float,
                 is_multi_labels: bool,
                 use_s2n: bool,
                 sub_node_num_layers: int = None,
                 sub_node_encoder_aggr: str = "sum",
                 subname: str = "default",
                 metrics=["micro_f1", "macro_f1"],
                 hp_metric=None,
                 use_bn: bool = False,
                 use_gn: bool = False,
                 use_skip: bool = False,
                 dropout_channels: float = 0.0,
                 dropout_edges: float = 0.0,
                 layer_kwargs: Dict[str, Any] = {},
                 given_datamodule: SubgraphDataModule = None,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["given_datamodule"])
        assert given_datamodule is not None
        self.given_datamodule = given_datamodule

        embedding_type, num_embedding_channels = "Embedding", given_datamodule.num_channels_global
        if self.dh.replace_x_with_wl4pattern:
            embedding_type = "UseRawFeature"
            num_embedding_channels = given_datamodule.num_channels_sub
        elif given_datamodule.embedding is not None:
            embedding_type = "Pretrained"
        self.node_emb = VersatileEmbedding(
            embedding_type=embedding_type,
            num_entities=given_datamodule.num_nodes_global,
            num_channels=num_embedding_channels,
            pretrained_embedding=given_datamodule.embedding,
        )
        if self.h.use_s2n:
            if self.h.sub_node_num_layers == 0:
                encoder, decoder = MyIdentity(), MyIdentity()
                in_channels = self.node_emb.num_channels
            else:
                kws = dict(num_layers=self.h.sub_node_num_layers,
                           hidden_channels=self.h.hidden_channels,
                           out_channels=self.h.hidden_channels,
                           activation=self.h.activation,
                           dropout=self.h.dropout_channels,
                           use_gn=self.h.use_gn,
                           activate_last=True)
                num_aggr = self.h.sub_node_encoder_aggr.count("-") + 1
                encoder = MLP(in_channels=given_datamodule.num_channels_global, **kws)
                decoder = MLP(in_channels=self.h.hidden_channels * num_aggr, **kws)
                in_channels = self.h.hidden_channels

            self.sub_node_encoder = DeepSets(encoder=encoder, decoder=decoder,
                                             aggr=self.h.sub_node_encoder_aggr)
            num_nodes = given_datamodule.test_data.num_nodes
            num_train_nodes = given_datamodule.train_data.num_nodes
            out_channels = given_datamodule.num_classes
        else:
            self.sub_node_encoder = None
            num_nodes = given_datamodule.num_nodes_global
            num_train_nodes = None
            in_channels = self.node_emb.num_channels
            if self.dh.replace_x_with_wl4pattern:
                out_channels = given_datamodule.num_classes
            else:
                out_channels = self.h.hidden_channels

        # If weighted edges are using, some models require special kwargs.
        if given_datamodule.h.s2n_is_weighted:
            if self.h.encoder_layer_name == "GATConv":
                layer_kwargs["edge_dim"] = 1

        if self.h.encoder_layer_name == "LINKX":
            self.encoder = InductiveLINKX(
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_channels=self.h.hidden_channels,
                out_channels=out_channels,
                num_layers=self.h.num_layers,
                dropout=self.h.dropout_channels,
                num_train_nodes=num_train_nodes,
                **self.h.layer_kwargs,  # num_edge_layers, num_node_layers
            )
        else:
            if isinstance(self.h.encoder_layer_name, (ListConfig, list)):
                if isinstance(self.h.num_layers, int):  # TODO: Remove HARD-CODED PARTS.
                    self.h.num_layers = [2 for _ in range(len(self.h.encoder_layer_name) - 1)] + [self.h.num_layers]
                __encoder_cls__ = GraphEncoderSequential
            else:
                __encoder_cls__ = GraphEncoder

            self.encoder = __encoder_cls__(
                self.h.encoder_layer_name,
                self.h.num_layers,
                in_channels=in_channels,
                hidden_channels=self.h.hidden_channels,
                out_channels=out_channels,
                activation=self.h.activation,
                use_bn=self.h.use_bn,
                use_gn=self.h.use_gn,
                use_skip=self.h.use_skip,
                dropout_channels=self.h.dropout_channels,
                dropout_edges=self.h.dropout_edges,
                activate_last=False,
                **self.h.layer_kwargs,
            )

        if self.h.use_s2n or self.dh.replace_x_with_wl4pattern:
            self.readout = None
        else:
            self.readout = Readout("sum", use_in_mlp=False, use_out_linear=True,
                                   hidden_channels=self.h.hidden_channels,
                                   out_channels=given_datamodule.num_classes)
        if not self.h.is_multi_labels:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(self.h.metrics, self.h.is_multi_labels)

    def forward(self, x=None, batch=None, sub_x=None, sub_batch=None,
                edge_index=None, edge_attr=None, adj_t=None, x_to_xs=None):

        if self.dh.replace_x_with_wl4pattern:
            return self.encoder(x, edge_index)  # edge_index is actually None.

        if sub_x is not None:
            sub_x = self.node_emb(sub_x)
            x = self.sub_node_encoder(sub_x, sub_batch)
        else:
            x = self.node_emb(x)

        edge_index = adj_t if adj_t is not None else edge_index
        x = self.encoder(x, edge_index, edge_attr)

        if not self.h.use_s2n:
            if x_to_xs is not None:  # for connected subgraphs
                x = x[x_to_xs]
            _, x = self.readout(x, batch)
        return x

    def step(self, batch: Data, batch_idx: int):
        step_kws = try_getattr(
            batch, ["x", "batch", "sub_x", "sub_batch", "edge_index", "edge_attr", "adj_t", "x_to_xs"])
        logits = self.forward(**step_kws)

        eval_mask = getattr(batch, "eval_mask", None)
        if eval_mask is not None:
            logits, y = logits[eval_mask], batch.y[eval_mask]
        else:
            y = batch.y
        loss = self.loss(logits, y)
        return {"loss": loss,
                "loss_sum": loss.item() * y.size(0), "size": y.size(0),
                "logits": logits, "ys": y}

    def training_step(self, batch: Data, batch_idx: int):
        return self.step(batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch: Data, batch_idx: int):
        return self.step(batch=batch, batch_idx=batch_idx)

    def test_step(self, batch: Data, batch_idx: int):
        return self.step(batch=batch, batch_idx=batch_idx)

    def epoch_end(self, prefix, output_as_dict):
        loss, loss_sum, size, logits, ys = try_get_from_dict(
            output_as_dict, ["loss", "loss_sum", "size", "logits", "ys"], as_dict=False)
        self.log(f"{prefix}/loss", sum(loss_sum) / sum(size), prog_bar=False)

        logits = torch.cat(logits)  # [*, C]
        ys = torch.cat(ys)  # [*] or [*, C]
        for metric, value in self.evaluator(logits, ys).items():
            self.log(f"{prefix}/{metric}", value, prog_bar=True)
            if prefix == "test" and self.h.hp_metric == metric:
                self.logger.log_metrics({"hp_metric": float(value)})

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end("train", ld_to_dl(outputs))

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end("valid", ld_to_dl(outputs))

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.epoch_end("test", ld_to_dl(outputs))

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(),
                                lr=self.h.learning_rate, weight_decay=self.h.weight_decay)


if __name__ == '__main__':

    def _pprint_tensor_dict(td: dict):
        _kv_dict = {}
        for k, v in td.items():
            if isinstance(v, torch.Tensor) and v.dim() > 1:
                _kv_dict[k] = f"Tensor({v.size()}, " \
                              f"mean={round(v.mean().item(), 4)}, " \
                              f"std={round(torch.std(v).item(), 4)})"
            else:
                _kv_dict[k] = v
        print(_kv_dict)


    NAME = "PPIBP"
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio

    PATH = "/mnt/nas2/GNN-DATA/SUBGRAPH"
    E_TYPE = "gin"  # gin, graphsaint_gcn, glass

    USE_S2N = True  # NOTE: important
    USE_SPARSE_TENSOR = False
    PRE_ADD_SELF_LOOPS = False
    SUBGRAPH_BATCHING = None if USE_S2N else "separated"  # separated, connected

    if USE_S2N:
        REPLACE_X_WITH_WL4PATTERN = False
    else:
        REPLACE_X_WITH_WL4PATTERN = False  # NOTE: important
    if REPLACE_X_WITH_WL4PATTERN:
        WL4PATTERN_ARGS = [0, "color"]  # color, cluster
    else:
        WL4PATTERN_ARGS = None

    ENCODER_NAME = "GCNConv"  # ["Linear", "GCNConv"]  # GATConv, LINKX, FAConv, GINConv
    NUM_LAYERS = 2
    if isinstance(ENCODER_NAME, list):
        NUM_LAYERS = [2, 3]

    if ENCODER_NAME == "GATConv":
        LAYER_KWARGS = {
            "edge_dim": 1,
            "add_self_loops": not PRE_ADD_SELF_LOOPS,
        }
    elif ENCODER_NAME == "LINKX":
        LAYER_KWARGS = {
            "num_edge_layers": 1,
            "num_node_layers": 1,
        }
    elif ENCODER_NAME == "FAConv":
        LAYER_KWARGS = {
            "eps": 0.2,
        }
    elif ENCODER_NAME == "GINConv":
        LAYER_KWARGS = {
            "train_eps": True,
        }
    else:
        LAYER_KWARGS = {}

    seed_everything(42)
    _sdm = SubgraphDataModule(
        dataset_name=NAME,
        dataset_path=PATH,
        embedding_type=E_TYPE,
        use_s2n=USE_S2N,
        edge_thres=0.0,
        use_consistent_processing=True,
        post_edge_normalize="standardize_then_trunc_thres_max_linear",
        post_edge_normalize_arg_1=0.0,
        post_edge_normalize_arg_2=2.0,
        s2n_target_matrix="adjacent_no_self_loops",
        s2n_is_weighted=True,
        subgraph_batching=SUBGRAPH_BATCHING,
        batch_size=32,
        eval_batch_size=5,
        use_sparse_tensor=USE_SPARSE_TENSOR,
        pre_add_self_loops=False,
        replace_x_with_wl4pattern=REPLACE_X_WITH_WL4PATTERN,
        wl4pattern_args=WL4PATTERN_ARGS,
        custom_splits=None,
    )
    _gnm = GraphNeuralModel(
        encoder_layer_name=ENCODER_NAME,
        num_layers=NUM_LAYERS,
        hidden_channels=64,
        activation="relu",
        learning_rate=0.001,
        weight_decay=1e-6,
        is_multi_labels=(NAME == "HPONeuro"),
        use_s2n=USE_S2N,
        sub_node_num_layers=0,
        use_bn=False,
        use_gn=True,
        use_skip=False,
        dropout_channels=0.0,
        dropout_edges=0.0,
        layer_kwargs=LAYER_KWARGS,
        given_datamodule=_sdm,
    )
    print(_gnm)
    for _i, _b in enumerate(_sdm.train_dataloader()):
        print(_b)
        _step_out = _gnm.training_step(_b, _i)
        _pprint_tensor_dict(_step_out)
        _gnm.training_epoch_end([_step_out, _step_out])
        break
    for _i, _b in enumerate(_sdm.val_dataloader()):
        _step_out = _gnm.validation_step(_b, _i)
        _pprint_tensor_dict(_step_out)
        _gnm.validation_epoch_end([_step_out, _step_out])
        break
    for _i, _b in enumerate(_sdm.test_dataloader()):
        _step_out = _gnm.test_step(_b, _i)
        _pprint_tensor_dict(_step_out)
        _gnm.test_epoch_end([_step_out, _step_out])
        break

    print("--- End")
