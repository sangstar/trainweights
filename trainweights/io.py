import os
from collections import defaultdict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import trainweights
from trainweights._C import TrainWeightsLoader
import json

SPLITTER_CHAR = '-'


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


class LazyReader:
    def __init__(self, filename):
        self._loader = TrainWeightsLoader(filename)

    def load_tensor(self, name) -> torch.Tensor:
        loaded = self._loader.load_tensor(name)
        return torch.from_dlpack(loaded[1])

def should_quantize(tensor_a: torch.Tensor, tensor_b: torch.Tensor, tol=1e-3):
    return torch.allclose(tensor_a, tensor_b, tol)




# loader = trainweights.io.LazyReader(filename)
# tens = loader.load_tensor("distilbert.embeddings.word_embeddings.weight")


def is_tensor_dict(d: dict) -> bool:
    if not isinstance(d, dict): return False
    for k, v in d.items():
        if not isinstance(v, torch.Tensor):
            return False
    return True

def json_serialize(d: dict, filename):
    # Serialize to string first - this will fail before touching the file
    json_str = json.dumps(d)

    with open(f"{filename}.json", "w") as f:
        f.write(json_str)

def is_json_serializable(d: dict) -> bool:
    try:
        json.dumps(d)
        return True
    except:
        return False

def is_mixed_tensor_dict(d: dict) -> bool:
    if not isinstance(d, dict): return False
    has_tensor = False
    has_non_tensor = False
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            has_tensor = True
        elif isinstance(v, dict):
            if is_mixed_tensor_dict(v):
                return True
        else:
            has_non_tensor = True
    return has_tensor and has_non_tensor

def save_mixed_param_dict(mixed: dict, fname, quantize_i8, failed_to_save):
    # collect non-tensor and tensor dicts
    non_tensor_entries = {}
    tensor_entries = {}
    for k, v in mixed.items():
        if isinstance(v, dict):
            save_mixed_param_dict(v)
        elif isinstance(v, torch.Tensor):
            tensor_entries[k] = v
        else:
            non_tensor_entries[k] = v
    dispatch_saver_on_obj(tensor_entries, fname, "", "tensors",  quantize_i8, failed_to_save)
    dispatch_saver_on_obj(non_tensor_entries, fname, "",  "notensors", quantize_i8, failed_to_save)

def save_tensor_dict(t_dict: dict, fname: str, quantize_i8):
    tensors = []
    names = []
    should_quant = quantize_i8 is True and "optimizer" not in fname

    for key, param in t_dict.items():
        if is_tensor_dict(param):
            save_tensor_dict(param, fname, quantize_i8)
        else:
            val = param.numpy()
            tensors.append(val)
            names.append(key)

    if names and tensors:
        filename = f"{fname}.tws"
        if should_quant:
            trainweights.quantize_and_save(filename, names, tensors)
        else:
            trainweights.save(filename, names, tensors)

def is_dict_of_containers(val):
    if not isinstance(val, dict): return False
    return all(isinstance(x, (dict, list)) for x in list(val.values()))

def get_file_prefix(key, prefix, name):
    if key:
        return f"{prefix}/{SPLITTER_CHAR.join([key, name])}".rstrip(SPLITTER_CHAR)
    else:
        return f"{prefix}/{name}"


def dispatch_saver_on_obj(val: object, key: str, prefix, name, quantize_i8, failed_to_save: list):

    fname = get_file_prefix(key, prefix, name)
    if is_mixed_tensor_dict(val):
        save_mixed_param_dict(val, fname, quantize_i8, failed_to_save)
    elif is_json_serializable(val):
        json_serialize(val, fname)
    elif is_tensor_dict(val):
        save_tensor_dict(val, fname, quantize_i8)
    elif is_dict_of_containers(val):
        for k, v in val.items():
            dispatch_saver_on_obj(v, f"{key}-{k}", prefix, name, quantize_i8, failed_to_save)
    elif isinstance(val, list):
        for idx, v in enumerate(val):
            dispatch_saver_on_obj(v, fname, str(idx), quantize_i8, failed_to_save)
    elif isinstance(val, (int, float, str, torch.Tensor)):
        dispatch_saver_on_obj({key : val}, "", prefix, name, quantize_i8, failed_to_save)
    else:
        failed_to_save.append((key, val))


def save_state(sd: dict, prefix, name, quantize_i8=False) -> list:
    failed_to_save = []
    for key, param in sd.items():
        dispatch_saver_on_obj(param, key, prefix, name, quantize_i8, failed_to_save)

    return failed_to_save


def save_checkpoint(ckpt_dict: dict, dirname: str, quantize_i8=False) -> None:
    faileds = []
    faileds += save_state(ckpt_dict, dirname, "", quantize_i8)
    print(faileds)



def save_model(model, filename, quantize_i8=False):
    save_tensor_dict(model.state_dict(), filename.replace(".tws", ""), quantize_i8)


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

def try_morph_dict(orig, subdict, levels):
    view = orig
    for level in levels:
        view = view[level]
    for key in subdict:
        if key in view:
            levels.append(key)
            return True, subdict[key]
    return False, subdict

def merge(orig, to_merge):
    levels = []
    subdict = to_merge
    while True:
        ok, subdict = try_morph_dict(orig, subdict, levels)
        if not ok:
            break
    current = orig
    for key in levels:
        current = current[key]
    current.update(current | subdict)



def commit_to_ckpt_dict(ckpt_dict, state_dict, filename):
    levels: list[str] = (filename
                         .replace(".tws", "")
                         .replace(".json", "")
                         .replace(f"{SPLITTER_CHAR}notensors", "")
                         .replace(f"{SPLITTER_CHAR}tensors", "")
                         .split(SPLITTER_CHAR))
    subdict = {}
    for idx, level in enumerate(reversed(levels)):
        if idx == 0:
            subdict = state_dict
        if level != "notensors" or level != "tensors":
            subdict = {level : subdict}
        else:
            print("spotted")
    merge(ckpt_dict, subdict)



def load_checkpoint(ckpt_dir):
    ckpt_dict = {}
    for file in os.listdir(ckpt_dir):
        filename = os.path.join(ckpt_dir, file)
        if filename.endswith(".tws"):
            names, dltensors = trainweights.load(filename)
            state_dict = {}
            for key, tensor in zip(names, dltensors):
                state_dict[key] = torch.from_dlpack(tensor)
            commit_to_ckpt_dict(ckpt_dict, state_dict, file)
        if filename.endswith(".json"):
            with open(filename, "r") as f:
                loaded = json.load(f)
                commit_to_ckpt_dict(ckpt_dict, loaded, file)
    return ckpt_dict
