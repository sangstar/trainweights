import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import trainweights

def save_model(model, filename):
    tensors = []
    names = []
    for key, param in model.state_dict().items():
        val = param.numpy()
        tensors.append(val)
        names.append(key)

    trainweights.quantize_and_save(filename, names, tensors)

def load_model(filename, model_id, device_str = "cpu"):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
    ).to(device_str)
    keys, dltensors = trainweights.load(filename)
    state_dict = {}
    for key, tensor in zip(keys, dltensors):
        state_dict[key] = torch.from_dlpack(tensor)

    model.load_state_dict(state_dict)
    return model
