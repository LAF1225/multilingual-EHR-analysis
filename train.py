import argparse
import json
import shutil
import time

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

from config import DATA, MAX_LEN, MODELS, MODELS_DIR
from utils import macro_f1


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS.keys(), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="effective batch = batch * grad_accum; keep it at 16")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--debug", action="store_true", help="train on 200 notes, 1 epoch")
    return p.parse_args()


def main():
    args = parse()
    set_seed(args.seed)
    name = MODELS[args.model]
    out = MODELS_DIR / f"{args.model}_seed{args.seed}"
    labels = json.load(open(DATA / "labels.json"))

    tok = AutoTokenizer.from_pretrained(name)

    def to_ds(csv, n=None):
        df = pd.read_csv(DATA / csv)
        if n:
            df = df.sample(n=min(n, len(df)), random_state=args.seed)
        ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}),
                                 preserve_index=False)
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                      batched=True, remove_columns=["text"])

    train_ds = to_ds("train.csv", 200 if args.debug else None)
    val_ds = to_ds("val.csv", 100 if args.debug else None)

    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
    )

    def compute_metrics(eval_pred):
        logits, y = eval_pred
        pred = logits.argmax(-1)
        return {"macro_f1": macro_f1(y, pred), "accuracy": accuracy_score(y, pred)}

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,            # skips optimizer state -> saves disk space
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=1 if args.debug else args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=20,
        seed=args.seed,
        fp16=torch.cuda.is_available(),  # mixed precision only on an NVIDIA GPU
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    val_metrics = trainer.evaluate()

    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    shutil.rmtree(out / "checkpoints", ignore_errors=True)

    info = {"model": args.model, "hf_id": name, "seed": args.seed, "train_minutes": round(minutes, 1),
            "val_macro_f1": val_metrics["eval_macro_f1"], "val_accuracy": val_metrics["eval_accuracy"],
            "args": vars(args), "device": str(trainer.args.device)}
    json.dump(info, open(out / "train_info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
