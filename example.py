import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import trainweights
print(trainweights.__file__)
MODEL_NAME = "distilbert-base-uncased"


# -----------------------------
# Checkpoint utilities
# -----------------------------
def build_checkpoint(
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=None,
        step=None,
        rng_state=True,
):
    ckpt = {
        "model": model.state_dict(),
    }

    if optimizer:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler:
        ckpt["scheduler"] = scheduler.state_dict()
    if scaler:
        ckpt["scaler"] = scaler.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    if step is not None:
        ckpt["step"] = step

    if rng_state:
        ckpt["rng"] = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }

    return ckpt


def restore_checkpoint(
        ckpt,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
):
    model.load_state_dict(ckpt["model"])

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    if "rng" in ckpt:
        torch.set_rng_state(ckpt["rng"]["torch"])
        if ckpt["rng"]["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])

    return ckpt.get("epoch", 0), ckpt.get("step", 0)


# -----------------------------
# Demo training
# -----------------------------
def main():
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    tensors = []
    names = []
    for key, param in model.state_dict().items():
        val = param.numpy()
        if len(val.shape) == 1:
            print(key, val[:5])
        else:
            print(key, val[0, :5])
        tensors.append(val)
        names.append(key)
    trainweights.quantize_and_save("trainweights_tensors.tws", names, tensors)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=20)

    scaler = torch.cuda.amp.GradScaler(enabled=False)

    texts = [
        "I love this movie",
        "This was terrible",
        "Amazing experience",
        "Worst product ever",
    ]
    labels = torch.tensor([1, 0, 1, 0])

    inputs = tokenizer(texts, padding=True, return_tensors="pt")

    model.train()

    for epoch in range(2):
        for step in range(10):
            optimizer.zero_grad()

            out = model(**inputs, labels=labels)
            loss = out.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

    print("Saving checkpoint...")

    ckpt = build_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=epoch,
        step=step,
    )

    torch.save(ckpt, "hf_checkpoint.pt")

    # -------------------------
    # Restore into fresh model
    # -------------------------
    print("Reloading...")

    model2 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    optimizer2 = AdamW(model2.parameters(), lr=5e-5)
    scheduler2 = LinearLR(optimizer2, start_factor=1.0, end_factor=0.1, total_iters=20)
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)

    loaded = torch.load("hf_checkpoint.pt", map_location=device)

    restore_checkpoint(
        loaded,
        model2,
        optimizer2,
        scheduler2,
        scaler2,
    )

    # sanity check
    with torch.no_grad():
        out1 = model(**inputs).logits
        out2 = model2(**inputs).logits

        diff = (out1 - out2).abs().max().item()

    print("Max diff after restore:", diff)


if __name__ == "__main__":
    main()
