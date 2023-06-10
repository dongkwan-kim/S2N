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

    def extra_repr(self) -> str:
        return f"(use_s2n_jk): {self.h.use_s2n_jk}"

    def __init__(self,
                 encoder_layer_name: Union[str, List[str]],
                 num_layers: Union[int, List[int]],
                 hidden_channels: int,
                 activation: str,
                 learning_rate: float,
                 weight_decay: float,
                 is_multi_labels: bool,
                 use_s2n: bool,
                 sub_node_encoder_name: str = "DeepSets",  # todo
                 sub_node_num_layers: int = None,
                 sub_node_encoder_aggr: str = "sum",
                 sub_node_encoder_layer_kwargs: Dict[str, Any] = {},
                 use_s2n_jk: str = "sum",
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
                 use_lr_scheduler=False,
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

        self.sub_node_set_encoder, self.sub_node_graph_encoder = None, None
        if self.h.use_s2n:
            sub_node_encoder_kwargs = dict(num_layers=self.h.sub_node_num_layers,
                                           hidden_channels=self.h.hidden_channels,
                                           out_channels=self.h.hidden_channels,
                                           activation=self.h.activation,
                                           activate_last=True)

            if self.h.sub_node_encoder_name == "DeepSets":
                if self.h.sub_node_num_layers == 0:
                    encoder, decoder = MyIdentity(), MyIdentity()
                    in_channels = self.node_emb.num_channels
                else:
                    sub_node_encoder_kwargs.update(dict(dropout=self.h.dropout_channels))
                    num_aggr = self.h.sub_node_encoder_aggr.count("-") + 1
                    encoder = MLP(in_channels=given_datamodule.num_channels_global, **sub_node_encoder_kwargs)
                    decoder = MLP(in_channels=self.h.hidden_channels * num_aggr, **sub_node_encoder_kwargs)
                    in_channels = self.h.hidden_channels

                self.sub_node_set_encoder = DeepSets(encoder=encoder, decoder=decoder,
                                                     aggr=self.h.sub_node_encoder_aggr)

            else:  # sub_node_encoder_name == "GCNConv", "GATConv", ...
                assert self.h.sub_node_num_layers > 0
                self.sub_node_graph_encoder = GraphEncoder(
                    self.h.sub_node_encoder_name,
                    in_channels=self.node_emb.num_channels,
                    use_bn=self.h.use_bn,
                    use_gn=self.h.use_gn,
                    use_skip=self.h.use_skip,
                    dropout_channels=self.h.dropout_channels,
                    dropout_edges=self.h.dropout_edges,
                    **sub_node_encoder_kwargs,
                    **self.h.sub_node_encoder_layer_kwargs,
                )
                in_channels = self.h.hidden_channels
                self.sub_node_set_encoder = DeepSets(encoder=MyIdentity(), decoder=MyIdentity(),
                                                     aggr=self.h.sub_node_encoder_aggr)

            if self.given_datamodule.num_channels_wl > 0:
                self.wl_encoder = nn.Linear(self.given_datamodule.num_channels_wl, in_channels)
                self.x_tmp_encoder = nn.Linear(in_channels, in_channels)  # todo:

            num_nodes = given_datamodule.test_data.num_nodes
            num_train_nodes = given_datamodule.train_data.num_nodes
        else:
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
                activate_last=True,
                **self.h.layer_kwargs,
            )

        self.readout, self.lin_last = None, None
        if self.h.use_s2n:
            assert (in_channels == out_channels) if self.h.use_s2n_jk == "sum" else True
            out_channels_total = (in_channels + out_channels) if self.h.use_s2n_jk == "concat" else out_channels
            self.lin_last = nn.Sequential(nn.Dropout(p=self.h.dropout_channels),
                                          nn.Linear(out_channels_total, given_datamodule.num_classes))
        elif not (self.h.use_s2n or self.dh.replace_x_with_wl4pattern):
            self.readout = Readout("sum", use_in_mlp=False, use_out_linear=True,
                                   hidden_channels=self.h.hidden_channels,
                                   out_channels=given_datamodule.num_classes)
        if not self.h.is_multi_labels:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
        self.evaluator = Evaluator(self.h.metrics, self.h.is_multi_labels)

    def forward(self,
                x=None, batch=None,
                sub_x=None, sub_batch=None,
                sub_x_weight=None, sub_x_wl=None, sub_edge_index=None,
                edge_index=None, edge_attr=None, adj_t=None, x_to_xs=None):

        if self.dh.replace_x_with_wl4pattern:
            return self.encoder(x, edge_index)  # edge_index is actually None.

        if sub_x is not None:
            sub_x = self.node_emb(sub_x)
            if self.sub_node_graph_encoder is not None:
                assert sub_edge_index is not None
                sub_x = self.sub_node_graph_encoder(sub_x, sub_edge_index, batch=sub_batch)
            x = self.sub_node_set_encoder(sub_x, batch=sub_batch, x_weight=sub_x_weight)

            # todo: refactoring
            if sub_x_wl is not None:
                # sum
                x = self.x_tmp_encoder(x) + self.wl_encoder(sub_x_wl)

                # cat
                # x = torch.cat([x, sub_x_wl], dim=1)  # [N, F] cat [N, F_wl] --> [N, F + F_wl]
        else:
            x = self.node_emb(x)

        edge_index = adj_t if adj_t is not None else edge_index
        if self.h.use_s2n_jk == "concat":
            x = torch.cat([x, self.encoder(x, edge_index, edge_attr)], dim=1)
        elif self.h.use_s2n_jk == "sum":
            x = x + self.encoder(x, edge_index, edge_attr)
        else:
            x = self.encoder(x, edge_index, edge_attr)

        if self.h.use_s2n:
            x = self.lin_last(x)
        else:
            if x_to_xs is not None:  # for connected subgraphs
                x = x[x_to_xs]
            _, x = self.readout(x, batch)
        return x

    def step(self, batch: Data, batch_idx: int):
        step_kws = try_getattr(
            batch, ["x", "batch", "sub_x", "sub_batch",
                    "sub_x_weight", "sub_x_wl", "sub_edge_index",
                    "edge_index", "edge_attr", "adj_t", "x_to_xs"])
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
        optimizer = torch.optim.Adam(params=self.parameters(),
                                     lr=self.h.learning_rate, weight_decay=self.h.weight_decay)
        # https://lightning.ai/docs/pytorch/1.5.4/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        if self.h.use_lr_scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, min_lr=5e-5),
                    "monitor": "valid/loss",
                    "frequency": 1,  # "frequency" should be set to a multiple of "trainer.check_val_every_n_epoch"
                },
            }
        else:
            return optimizer


if __name__ == '__main__':

    from termcolor import cprint


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
    NUM_LAYERS = 1
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

    SUB_NODE_ENCODER_NAME = "DeepSets"  # DeepSets, GCNConv
    if SUB_NODE_ENCODER_NAME == "DeepSets":
        SUB_NODE_NUM_LAYERS = 0
        USE_SUB_EDGE_INDEX = True
    else:
        SUB_NODE_NUM_LAYERS = 2
        USE_SUB_EDGE_INDEX = True

    seed_everything(42)
    _sdm = SubgraphDataModule(
        dataset_name=NAME,
        dataset_path=PATH,
        embedding_type=E_TYPE,
        use_s2n=USE_S2N,
        s2n_mapping_matrix_type="unnormalized",
        s2n_set_sub_x_weight="original_sqrt_d_node_div_d_sub",
        s2n_use_sub_edge_index=USE_SUB_EDGE_INDEX,
        s2n_add_sub_x_wl=False,
        edge_thres=0.0,
        use_consistent_processing=True,
        post_edge_normalize="standardize_then_trunc_thres_max_linear",
        post_edge_normalize_arg_1=2.0,
        post_edge_normalize_arg_2=2.0,
        s2n_target_matrix="adjacent_with_self_loops",
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
        sub_node_encoder_name=SUB_NODE_ENCODER_NAME,
        sub_node_num_layers=SUB_NODE_NUM_LAYERS,
        use_s2n_jk="sum",
        use_bn=True,
        use_gn=False,
        use_skip=True,
        dropout_channels=0.2,
        dropout_edges=0.0,
        layer_kwargs=LAYER_KWARGS,
        given_datamodule=_sdm,
    )
    cprint("\n" + "-" * 50, "green")
    for _name, _item in _gnm.named_modules():
        print(_name)
        print(_item)
        cprint("-" * 10, "green")
    cprint("\n" + "-" * 50, "green")

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
