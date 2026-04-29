import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
import numpy as np


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_DIR = os.environ.get(
    "MMSA_DATASET_DIR",
    os.path.join(PROJECT_DIR, "..", "data"),
)
SUPPORTED_DATASETS = {"mosi", "mosei", "sims"}
GLOBAL_DOMAIN_TO_ID = {"mosei": 0, "mosi": 1, "sims": 2}


def _normalize_dataset_name(name):
    dataset_name = name.lower().strip()
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{name}'. Expected one of: {sorted(SUPPORTED_DATASETS)}."
        )
    return dataset_name


def _resolve_dataset_path(dataset_name, hyper):
    dataset_name = _normalize_dataset_name(dataset_name)

    base_or_file = hyper.datapath.strip() if isinstance(hyper.datapath, str) else ""
    if not base_or_file:
        base_or_file = DEFAULT_DATASET_DIR

    if os.path.isdir(base_or_file):
        data_path = os.path.join(base_or_file, f"{dataset_name}.pkl")
    else:
        data_path = base_or_file

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset file not found. "
            f"Expected: {data_path}. "
            "You can pass --datapath as a dataset directory or a concrete .pkl file path."
        )

    return data_path


def _load_single_domain(dataset_name, split, hyper):
    data_path = _resolve_dataset_path(dataset_name, hyper)
    with open(data_path, "rb") as file:
        data = pickle.load(file)

    split_data = data[split]
    labels = split_data["regression_labels"]
    if dataset_name == "sims":
        labels = labels * 3

    return {
        "text": split_data["text"],
        "audio": split_data["audio"],
        "vision": split_data["vision"],
        "regression_labels": labels,
    }


def _get_source_and_target_datasets(hyper):
    source_datasets = getattr(hyper, "source_datasets", None)
    target_dataset = getattr(hyper, "target_dataset", None)

    # Backward compatibility: if old --dataset is used, keep single-domain behavior.
    if source_datasets is None or len(source_datasets) == 0:
        fallback_dataset = _normalize_dataset_name(getattr(hyper, "dataset", "mosi"))
        source_datasets = [fallback_dataset]
    else:
        source_datasets = [_normalize_dataset_name(ds) for ds in source_datasets]

    if target_dataset is None or str(target_dataset).strip() == "":
        target_dataset = source_datasets[0]
    target_dataset = _normalize_dataset_name(target_dataset)

    return source_datasets, target_dataset


class MSADataset(Dataset):
    def __init__(self, hyper, split):
        source_datasets, target_dataset = _get_source_and_target_datasets(hyper)

        self.split = split

        if split in ["train", "valid"]:
            active_domains = source_datasets
        elif split == "test":
            active_domains = [target_dataset]
        else:
            raise ValueError(f"Unsupported split '{split}'.")

        self.samples = []

        self.orig_dims = None
        for domain_name in active_domains:
            domain_split_data = _load_single_domain(domain_name, split, hyper)

            current_dims = [
                domain_split_data["text"][0].shape[1],
                domain_split_data["audio"][0].shape[1],
                domain_split_data["vision"][0].shape[1],
            ]
            if self.orig_dims is None:
                self.orig_dims = current_dims
            elif self.orig_dims != current_dims:
                raise ValueError(
                    "All domains must share identical modality dimensions for DG training. "
                    f"Expected {self.orig_dims}, but got {current_dims} in domain '{domain_name}'."
                )

            domain_size = domain_split_data["audio"].shape[0]
            domain_id = GLOBAL_DOMAIN_TO_ID[domain_name]
            for idx in range(domain_size):
                self.samples.append(
                    {
                        "idx": idx,
                        "audio": domain_split_data["audio"][idx],
                        "vision": domain_split_data["vision"][idx],
                        "text": domain_split_data["text"][idx],
                        "label": domain_split_data["regression_labels"][idx],
                        "domain_label": domain_id,
                    }
                )

        if self.orig_dims is None:
            raise ValueError("No data found after loading source/target domains.")

    def get_dim(self):
        return self.orig_dims

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = np.asarray(sample["label"]).squeeze()
        return {
            "idx": torch.tensor(idx).long(),
            "audio": torch.tensor(sample["audio"]).float(),
            "vision": torch.tensor(sample["vision"]).float(),
            "text": torch.tensor(sample["text"]).float(),
            "label": torch.tensor(label).float(),
            "domain_label": torch.tensor(sample["domain_label"]).long(),
        }


def msa_collate_fn(batch):
    text_list = [item["text"] for item in batch]
    audio_list = [item["audio"] for item in batch]
    vision_list = [item["vision"] for item in batch]

    text = pad_sequence(text_list, batch_first=True, padding_value=0.0)
    audio = pad_sequence(audio_list, batch_first=True, padding_value=0.0)
    vision = pad_sequence(vision_list, batch_first=True, padding_value=0.0)

    labels = torch.stack([item["label"] for item in batch])
    domain_labels = torch.stack([item["domain_label"] for item in batch])
    indices = torch.stack([item["idx"] for item in batch])

    lengths = torch.tensor([item["text"].shape[0] for item in batch]).long()

    return {
        "idx": indices,
        "audio": audio,
        "vision": vision,
        "text": text,
        "label": labels,
        "domain_label": domain_labels,
        "lengths": lengths,
    }


def getdataloader(args):
    dataloaders = {}
    for split in ["train", "valid", "test"]:
        datasets = MSADataset(args, split=split)
        dataloaders[split] = DataLoader(
            datasets,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=msa_collate_fn,
        )

    orig_dim = datasets.get_dim()

    return dataloaders, orig_dim
