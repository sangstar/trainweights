import os

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import trainweights
from trainweights.io import build_checkpoint, restore_checkpoint

MODEL_NAME = "distilbert-base-uncased"


def main():
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    pwd = os.getenv("PWD") or None
    if pwd is None:
        raise RuntimeError("No save dir was set")
    else:
        save_dir = f"{pwd}/model"

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

    step = 0

    model.train()

    def forward_two_epochs(initial_step):
        for epoch in range(2):
            for step in range(initial_step, initial_step + 10):
                optimizer.zero_grad()

                out = model(**inputs, labels=labels)
                loss = out.loss

                loss.backward()
                optimizer.step()
                scheduler.step()
        return initial_step + 10

    step = forward_two_epochs(step)

    ckpt = build_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=2,
        step=step,
    )

    ckpt_dir = f"{save_dir}_{step}"
    os.mkdir(ckpt_dir)

    # Retain precision on original checkpoint
    trainweights.io.save_checkpoint(ckpt, ckpt_dir, quantize_i8=False)

    # Simulate another few training steps
    step = forward_two_epochs(step)

    ckpt = build_checkpoint(
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=2,
        step=step,
    )

    ckpt_dir = f"{save_dir}_{step}"
    os.mkdir(ckpt_dir)

    # Quantize this checkpoint at rest to save on disk usage
    trainweights.io.save_checkpoint(ckpt, ckpt_dir, quantize_i8=False)


    # Let's now load this checkpoint and compare them
    ckpt_dict_loaded = trainweights.io.load_checkpoint(ckpt_dir)

    print("Reloading...")

    model2 = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    optimizer2 = AdamW(model2.parameters(), lr=5e-5)
    scheduler2 = LinearLR(optimizer2, start_factor=1.0, end_factor=0.1, total_iters=20)
    scaler2 = torch.cuda.amp.GradScaler(enabled=False)

    restore_checkpoint(
        ckpt_dict_loaded,
        model2,
        optimizer2,
        scheduler2,
        scaler2,
    )

    with torch.no_grad():
        out1 = model(**inputs).logits
        out2 = model2(**inputs).logits

        diff = (out1 - out2).abs().max().item()

    print("Max diff after restore:", diff)


if __name__ == "__main__":
    main()