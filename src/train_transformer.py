"""Fine-tune DistilBERT / RoBERTa with a class-weight-aware Hugging Face Trainer (paper, Sec. 5.3-5.5)
and optional multi-label classification (Improvement 7).

Usage (Colab T4 is enough):
    python -m src.train_transformer --model distilbert --strategy baseline --seeds 42 43 44
    python -m src.train_transformer --model roberta --strategy class_weight
    python -m src.train_transformer --model distilbert --tune        # lr x batch-size grid on val macro-F1
    python -m src.train_transformer --model distilbert --multilabel  # multi-label emotion model
"""
from __future__ import annotations

import argparse
import inspect
import itertools
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from src.datasets import TEXT_COLUMN, TransformerDataset, load_splits
from src.evaluation import hf_compute_metrics, report_run
from src.imbalance import prepare_training_data
from src.utils import (ID2LABEL, LABEL2ID, LABELS, NUM_CLASSES, get_logger, load_config, model_dir, save_json, set_seed)

log = get_logger("train_transformer")
TEXT_COL = TEXT_COLUMN["transformer"]


class WeightedTrainer(Trainer):
    """Cross-entropy with per-class weights w_c = N / (|E| n_c) (paper, Sec. 5.4)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


class MultiLabelTrainer(Trainer):
    """BCE with logits loss for multi-label classification (Improvement 7)."""

    def __init__(self, *args, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        w = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=w)
        return (loss, outputs) if return_outputs else loss


def hf_compute_metrics_multilabel(eval_pred) -> dict:
    """compute_metrics callback for multi-label training."""
    from sklearn.metrics import f1_score, hamming_loss
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    preds = (probs >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "hamming_loss": float(hamming_loss(labels, preds)),
    }


def tune_multilabel_thresholds(y_true: np.ndarray, y_probs: np.ndarray, step: float = 0.05) -> np.ndarray:
    """Find per-class thresholds in [0.1, 0.9] maximizing per-class F1 on validation set."""
    from sklearn.metrics import f1_score
    best_thresh = np.full(y_probs.shape[1], 0.5)
    candidates = np.arange(0.10, 0.91, step)
    for c in range(y_probs.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        yt = y_true[:, c]
        yp = y_probs[:, c]
        if yt.sum() == 0:
            continue
        for t in candidates:
            pred = (yp >= t).astype(int)
            score = f1_score(yt, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)
        best_thresh[c] = best_t
    return best_thresh


def evaluate_multilabel(y_true: np.ndarray, y_probs: np.ndarray, thresholds: np.ndarray | None = None) -> dict:
    from sklearn.metrics import f1_score, hamming_loss
    if thresholds is None:
        thresholds = np.full(y_probs.shape[1], 0.5)
    else:
        thresholds = np.asarray(thresholds)
    preds = (y_probs >= thresholds).astype(int)
    micro_f1 = float(f1_score(y_true, preds, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    h_loss = float(hamming_loss(y_true, preds))
    per_class_f1 = f1_score(y_true, preds, average=None, zero_division=0)
    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "hamming_loss": h_loss,
        "per_class": {LABELS[i]: float(per_class_f1[i]) for i in range(len(LABELS))},
        "thresholds": {LABELS[i]: float(thresholds[i]) for i in range(len(LABELS))},
    }


def _training_args(cfg: dict, out_dir: str, hp: dict, seed: int) -> TrainingArguments:
    tc = cfg["transformer"]
    sig = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in sig else "evaluation_strategy"  # renamed in transformers 4.41
    kwargs = dict(
        output_dir=out_dir, num_train_epochs=hp["epochs"], learning_rate=hp["lr"],
        per_device_train_batch_size=hp["batch_size"], per_device_eval_batch_size=128,
        weight_decay=tc["weight_decay"], warmup_ratio=tc["warmup_ratio"], lr_scheduler_type="linear",
        save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, seed=seed, data_seed=seed,
        fp16=torch.cuda.is_available(), report_to="none", logging_steps=100,
    )
    kwargs[eval_key] = "epoch"
    return TrainingArguments(**kwargs)


def _run_one(cfg, name, checkpoint, strategy, hp, seed, train_df, val_df, weights, out_dir, multilabel: bool = False):
    tc = cfg["transformer"]
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    problem_type = "multi_label_classification" if multilabel else "single_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=NUM_CLASSES, problem_type=problem_type, id2label=ID2LABEL, label2id=LABEL2ID)
    tr = TransformerDataset(train_df, tok, TEXT_COL, tc["max_len"])
    va = TransformerDataset(val_df, tok, TEXT_COL, tc["max_len"])

    if multilabel:
        trainer = MultiLabelTrainer(
            model=model, args=_training_args(cfg, str(out_dir / "checkpoints"), hp, seed),
            train_dataset=tr, eval_dataset=va, data_collator=DataCollatorWithPadding(tok),
            compute_metrics=hf_compute_metrics_multilabel,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=tc["early_stopping_patience"])],
        )
    else:
        cw = torch.tensor([weights.get(c, 1.0) for c in range(NUM_CLASSES)], dtype=torch.float) if weights else None
        trainer = WeightedTrainer(
            model=model, args=_training_args(cfg, str(out_dir / "checkpoints"), hp, seed),
            train_dataset=tr, eval_dataset=va, data_collator=DataCollatorWithPadding(tok),
            compute_metrics=hf_compute_metrics, class_weights=cw,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=tc["early_stopping_patience"])],
        )
    trainer.train()
    val_f1 = trainer.evaluate()["eval_macro_f1"]
    return trainer, tok, val_f1


def train_transformer(cfg: dict, name: str, strategy: str = "baseline", seed: int | None = None,
                      tune: bool = False, splits: dict | None = None, save: bool = True,
                      multilabel: bool = False) -> dict:
    tc = cfg["transformer"]
    checkpoint = tc["models"][name]
    seed = cfg["project"]["seed"] if seed is None else seed
    splits = splits or load_splits(cfg, multilabel=multilabel)

    if multilabel:
        train_df, weights = splits["train"], None
        eff_strategy = "multilabel"
    else:
        train_df, weights = prepare_training_data(splits["train"], strategy, cfg, seed)
        eff_strategy = strategy

    val_df, test_df = splits["val"], splits["test"]
    out_dir = model_dir(cfg, name, eff_strategy, seed, create=True)

    base = {"lr": tc["lr"], "batch_size": tc["batch_size"], "epochs": tc["epochs"]}
    combos = [base]
    if tune:
        combos = [{"lr": lr, "batch_size": bs, "epochs": tc["epochs"]}
                  for lr, bs in itertools.product(tc["grid"]["lr"], tc["grid"]["batch_size"])]

    best = (-1.0, None, None, None)
    for hp in combos:
        log.info("Fine-tuning %s | strategy=%s | seed=%d | %s", name, eff_strategy, seed, hp)
        trainer, tok, f1 = _run_one(cfg, name, checkpoint, eff_strategy, hp, seed, train_df, val_df, weights, out_dir,
                                    multilabel=multilabel)
        log.info("val macro-F1 = %.4f", f1)
        if f1 > best[0]:
            best = (f1, hp, trainer, tok)
    best_f1, best_hp, trainer, tok = best

    te = TransformerDataset(test_df, tok, TEXT_COL, tc["max_len"])

    if multilabel:
        va = TransformerDataset(val_df, tok, TEXT_COL, tc["max_len"])
        val_pred = trainer.predict(va)
        val_probs = 1.0 / (1.0 + np.exp(-np.clip(val_pred.predictions, -50, 50)))
        thresholds = tune_multilabel_thresholds(val_pred.label_ids, val_probs)
        log.info("Tuned per-class thresholds: %s",
                 {LABELS[i]: round(float(thresholds[i]), 2) for i in range(len(LABELS))})

        test_pred = trainer.predict(te)
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_pred.predictions, -50, 50)))
        metrics = evaluate_multilabel(test_pred.label_ids, test_probs, thresholds)
        log.info("Test results (multi-label): micro-F1=%.4f macro-F1=%.4f hamming-loss=%.4f",
                 metrics["micro_f1"], metrics["macro_f1"], metrics["hamming_loss"])

        if save:
            trainer.save_model(str(out_dir))
            tok.save_pretrained(str(out_dir))
            save_json({"model": name, "checkpoint": checkpoint, "strategy": "multilabel", "seed": seed,
                       "hparams": best_hp, "val_macro_f1": best_f1, "max_len": tc["max_len"],
                       "metrics": metrics}, out_dir / "info.json")
            save_json(metrics["thresholds"], out_dir / "thresholds.json")
            save_json(metrics, out_dir / "multilabel_test_metrics.json")
        shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
        return {"val_macro_f1": best_f1, "hparams": best_hp, "out_dir": str(out_dir), "metrics": metrics}

    pred = trainer.predict(te)
    y_pred = np.argmax(pred.predictions, axis=-1)
    report_run(cfg, name, eff_strategy, seed, pred.label_ids, y_pred, split="test")

    if save:
        trainer.save_model(str(out_dir))
        tok.save_pretrained(str(out_dir))
        save_json({"model": name, "checkpoint": checkpoint, "strategy": eff_strategy, "seed": seed,
                   "hparams": best_hp, "val_macro_f1": best_f1, "max_len": tc["max_len"]}, out_dir / "info.json")
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)  # keep Drive/disk usage small
    return {"val_macro_f1": best_f1, "hparams": best_hp, "out_dir": str(out_dir)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["distilbert", "roberta"])
    ap.add_argument("--strategy", default="baseline")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--multilabel", action="store_true", help="train multi-label emotion model (Improvement 7)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    splits = load_splits(cfg, multilabel=a.multilabel)
    for s in a.seeds or cfg["seeds"]:
        train_transformer(cfg, a.model, a.strategy, s, a.tune, splits, multilabel=a.multilabel)


if __name__ == "__main__":
    main()
