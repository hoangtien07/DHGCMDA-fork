"""Train Stage 4 controls and Stage 5 candidates on a frozen outer fold."""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData

from clean_baselines import evaluate, take
from clean_protocol import DATASETS, RUNS, SEEDS, fit_views, load_frozen_fold
from prepare_v32 import DATA, DATASET, PORT, RELEASE, replace_once, sha
from run_v20 import train_full_fold
from score_predictions import type_metrics

CONTROLS = ("legacy", "gip-only", "association-only", "validation-only", "multilabel-only")
CANDIDATES = ("clean", "balanced", "bilinear", "simplified")
MULTILABEL = ("multilabel-only", "clean-multilabel")


def configure_model(author, args, variant):
    model = author.HeterogenousGraphCLAMIR(args.mi_num, args.dis_num, [256, 256], 64, args)
    if variant in {"balanced", "bilinear", "simplified"}:
        model.register_parameter("loss_log_scales", nn.Parameter(torch.zeros(3)))

    if variant in {"bilinear", "simplified"}:
        predictor = model.association_predictor

        class FullBilinear(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.mi_projector = base.mi_projector
                self.dis_projector = base.dis_projector
                self.temperature = base.temperature
                self.exist_relation = base.exist_relation
                self.type_relations = nn.Parameter(torch.diag_embed(base.type_relations.detach()))

            def forward(self, mi, disease):
                mi = self.mi_projector(mi)
                disease = self.dis_projector(disease)
                exists = (mi * self.exist_relation @ disease.T).sigmoid()
                logits = torch.einsum("mh,thk,dk->mdt", mi, self.type_relations, disease)
                probabilities = F.softmax(logits / self.temperature.clamp(0.5, 5.0), dim=-1)
                return torch.cat((exists.unsqueeze(-1), probabilities), dim=-1)

        model.association_predictor = FullBilinear(predictor)
    if variant == "simplified":
        # Node transforms and the typed predictor still run with zero HGT layers.
        model.hgt_layers = nn.ModuleList()
        model.enable_inter_view_cl = False
    return model.to(author.device)


def control_inputs(dataset, labels, semantic, split, variant):
    views = fit_views(labels, split["fit_pos"], semantic)
    classes = labels.shape[2]
    code = np.arange(1, classes + 1)
    legacy = variant in CONTROLS
    association_labels = labels if legacy and variant != "association-only" else views["masked_labels"]
    association = (association_labels * code).max(axis=2)
    matrix_dir = RELEASE / dataset if dataset != DATASET else DATA / "standard" / DATASET
    if legacy:
        disease = np.loadtxt(matrix_dir / "D_SSM2.txt")
        functional = np.loadtxt(matrix_dir / "M_FSM.txt")
        mirna = views["mirna_gip"] if variant == "gip-only" else np.loadtxt(matrix_dir / "M_GSM.txt")
    else:
        disease, functional, mirna = (views["disease_gip"], views["mirna_functional"], views["mirna_gip"])

    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32)
    semantic = tensor(semantic if legacy else views["disease_semantic"])
    association = tensor(association)
    # The released trainer reads fields 0–4 and 8–11. The callback handles holdout predictions.
    return [semantic, tensor(functional),
            [torch.as_tensor(split["fit_pos"]), torch.as_tensor(split["fit_neg"])],
            None, association, None, None, None,
            tensor(disease), tensor(mirna), semantic, tensor(functional)]


def adapt_trainer(author, model, masked_labels, variant, save_validation):
    raw = inspect.getsource(author.train_epoch_optimized)
    tail = raw.index("    # ✅ 修复: 在训练循环完成后进行测试")
    source = raw[:tail]
    source = replace_once(
        source, "        optim.step()",
        "        optim.step()\n        save_validation(epoch, concat_mi_tensor_view1, concat_dis_tensor_view1, "
        "G_mi_view1, G_mi_view2, G_dis_view1, G_dis_view2, hetero_data)")
    source = replace_once(
        source, "        if torch.isnan(tol_loss) or torch.isnan(mi_cl_loss) or torch.isnan(dis_cl_loss):",
        "        tol_loss = combine_loss(tol_loss, recover_loss, mi_cl_loss, dis_cl_loss, "
        "mi_recon_loss, dis_recon_loss, reg_loss)\n"
        "        if not torch.isfinite(tol_loss):\n            raise RuntimeError('Non-finite training loss')\n"
        "        if torch.isnan(tol_loss) or torch.isnan(mi_cl_loss) or torch.isnan(dis_cl_loss):")

    def combine_loss(original, recover, mi_cl, dis_cl, mi_recon, dis_recon, regularization):
        if not hasattr(model, "loss_log_scales"):
            return original
        terms = [recover, 0.15 * (mi_recon + dis_recon)]
        if variant != "simplified":
            terms.append(mi_cl + dis_cl)
        values = torch.stack(terms)
        log_scales = model.loss_log_scales[:len(terms)].clamp(-3, 3)
        return (values * torch.exp(-log_scales) + log_scales).sum() + 0.0001 * regularization

    criterion = author.SimplifiedMultiTypeAssociationLoss
    if variant in MULTILABEL:
        targets = torch.as_tensor(masked_labels, dtype=torch.float32, device=author.device)

        class MultiLabelLoss(criterion):
            def _compute_type_loss(self, one_index, type_pred, scalar_targets):
                pairs = self._process_indices(one_index, *type_pred.shape[:2])
                labels = targets[pairs[:, 0], pairs[:, 1]]
                scores = type_pred[pairs[:, 0], pairs[:, 1]].clamp_min(1e-7)
                normalized = labels / labels.sum(1, keepdim=True)
                return -(normalized * scores.log() * self.class_weights).sum(1).mean()

        criterion = MultiLabelLoss
    namespace = dict(vars(author), save_validation=save_validation, combine_loss=combine_loss,
                     SimplifiedMultiTypeAssociationLoss=criterion)
    exec(compile(source, author.__file__, "exec"), namespace)
    return namespace["train_epoch_optimized"]


def model_args(labels):
    return SimpleNamespace(mi_num=labels.shape[0], dis_num=labels.shape[1],
                           num_association_types=labels.shape[2], n_head=8, nlayer=2,
                           dropout=0.3, alpha=0.5, epoch=650)


def collect(model, inputs, pairs):
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            score = model(*inputs)[0]
            selected = score[pairs[:, 0], pairs[:, 1]].detach().cpu().numpy()
        expected = model.association_predictor.type_relations.shape[0] + 1
        if selected.shape != (len(pairs), expected) or not np.isfinite(selected).all():
            raise RuntimeError("Model returned invalid predictions")
        return selected
    finally:
        model.train(previous)


def run_directory(options, variant):
    return RUNS / "clean_models" / f"{options.dataset}-{options.seed}-fold{options.fold}-{variant}-{options.device}"


def train_candidate(options, author, args, labels, semantic, split, split_hash):
    variant = options.variant
    output = run_directory(options, variant)
    output.mkdir(parents=True, exist_ok=False)
    model = configure_model(author, args, variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    val = np.concatenate((split["val_pos"], split["val_neg"]))
    val_truth = take(labels, split["val_pos"])
    best = {"score": -1.0}
    select_epoch = variant not in CONTROLS or variant == "validation-only"

    def checkpoint(epoch, *inputs):
        if epoch != args.epoch and (not select_epoch or epoch % 50):
            return
        scores = collect(model, inputs, val)
        metrics = type_metrics(val_truth, scores[:len(val_truth), 1:])
        value = metrics["tdrc_top1_f1"]
        if value > best["score"]:
            best.update(score=value, epoch=epoch, metrics=metrics)
            graph = {key: tensor.detach().cpu() for key, tensor in inputs[-1].edge_index_dict.items()}
            checkpoint = {
                "state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "inputs": [tensor.detach().cpu() for tensor in inputs[:-1]],
                "graph": graph, "validation_scores": torch.from_numpy(scores),
            }
            torch.save(checkpoint, output / "best.tmp")
            (output / "best.tmp").replace(output / "best.pt")

    data = control_inputs(options.dataset, labels, semantic, split, variant)
    fit_labels = np.zeros_like(labels)
    fit_labels[tuple(split["fit_pos"].T)] = take(labels, split["fit_pos"])
    train = adapt_trainer(author, model, fit_labels, variant, checkpoint)
    _, steps = train_full_fold(train, model, data, optimizer, args)
    state = output / "best.pt"
    if not state.is_file():
        raise RuntimeError("Training produced no validation checkpoint")
    record = {"run": vars(options), "split_sha256": split_hash, "best_epoch": best["epoch"],
              "optimizer_steps": steps,
              "inner_validation": best["metrics"], "checkpoint_sha256": sha(state)}
    (output / "ready.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


def confirm_candidate(options, author, args, labels, split, split_hash):
    variant = options.variant
    if variant in CANDIDATES and variant != "clean":
        # Eligibility is based only on the four validation results for this fold.
        selected = []
        for candidate in CANDIDATES:
            record = json.loads((run_directory(options, candidate) / "ready.json").read_text(encoding="utf-8"))
            if record["split_sha256"] != split_hash:
                raise RuntimeError(f"Validation used a different split: {candidate}")
            selected.append(record["inner_validation"]["tdrc_top1_f1"])
        winner = CANDIDATES[int(np.argmax(selected))]
        if variant != winner:
            raise RuntimeError(f"Inner validation selected {winner}; cannot confirm {variant}")
    output = run_directory(options, variant)
    ready = json.loads((output / "ready.json").read_text(encoding="utf-8"))
    state = output / "best.pt"
    if ready["split_sha256"] != split_hash or sha(state) != ready["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from its validation record")
    if (output / "completion.json").exists():
        raise RuntimeError(f"Outer-test result already exists at {output}")
    checkpoint = torch.load(state, weights_only=True)
    model = configure_model(author, args, variant)
    model.load_state_dict(checkpoint["state"])
    graph = HeteroData()
    for kind, index in checkpoint["graph"].items():
        graph[kind].edge_index = index
    pairs = np.concatenate((split["test_pos"], split["test_neg"]))
    test = collect(model, (*checkpoint["inputs"], graph), pairs)
    scores = np.concatenate((checkpoint["validation_scores"].numpy(), test))
    on_test = variant in CONTROLS and variant != "validation-only"
    result = evaluate(labels, split, scores[:, 1:], scores[:, 0], threshold_on_test=on_test)
    np.savez_compressed(output / "test_scores.npz", scores=test, pairs=pairs)
    result.update({"run": vars(options), "split_sha256": split_hash, "best_epoch": ready["best_epoch"],
                   "scores_sha256": sha(output / "test_scores.npz")})
    (output / "completion.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fit", "confirm"))
    parser.add_argument("variant", choices=CONTROLS + CANDIDATES + ("clean-multilabel",))
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--fold", choices=range(1, 6), type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    options = parser.parse_args()
    if (options.device == "cuda") != torch.cuda.is_available():
        raise RuntimeError("Requested device differs from PyTorch availability; hide CUDA to run on CPU")
    if options.variant in MULTILABEL and options.dataset != DATASET:
        raise RuntimeError("The multilabel variant is registered for v3.2")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    labels, semantic, split, split_hash = load_frozen_fold(options.dataset, options.seed, options.fold)
    sys.path.insert(0, str(PORT if options.dataset == DATASET else RELEASE))
    import main_experiments_hetero1 as author

    author.seed_torch(options.seed)
    args = model_args(labels)
    if options.mode == "fit":
        train_candidate(options, author, args, labels, semantic, split, split_hash)
    else:
        confirm_candidate(options, author, args, labels, split, split_hash)


if __name__ == "__main__":
    main()
