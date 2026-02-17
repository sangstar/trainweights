import os

from transformers import AutoModelForSequenceClassification
import trainweights

MODEL_NAME = "distilbert-base-uncased"

def main():
    device = "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    dir_name = ""
    default_dir = os.getenv("PWD") or None
    save_dir = dir_name or default_dir

    if save_dir is None:
        raise RuntimeError("No save dir was set")

    model.save_pretrained(save_dir)

    safetensors_filepath = f"{save_dir}/model.safetensors"
    trainweights_filepath = f"{save_dir}/trainweights_tensors.tws"

    trainweights.save_model(model, trainweights_filepath)

    print(f"Safetensors filesize: "
          f"{os.stat(safetensors_filepath).st_size / 1e9:.2f} GB")

    print(f"Trainweights int8 quantized filesize: "
          f"{os.stat(trainweights_filepath).st_size / 1e9:.2f} GB")

    tw_model = trainweights.load_model(trainweights_filepath, MODEL_NAME)

    num_elems = 5
    state_dicts = zip(list(model.state_dict().items())[:num_elems], list(tw_model.state_dict().items())[:num_elems])

    for (orig_key, orig_value), (tw_key, tw_value) in state_dicts:
        orig_mean = orig_value.mean()
        mean_diff = ((orig_value - tw_value).mean() / orig_mean).item()
        print(f"Proportioned mean difference for key {orig_key}: {mean_diff:.3f}")

if __name__ == "__main__":
    main()