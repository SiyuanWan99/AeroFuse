
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.io
import torch


ROOT = Path(__file__).resolve().parent
from aerofuse_model import MFT_ConformerSDHSoftAnchor


DATASETS = {
    "Houston": {
        "display_name": "Houston2013",
        "hsi_channels": 144,
        "aux_channels": 1,
        "classes": 15,
        "sdh_rank": 24,
        "parameters": 708931,
        "aux_file": "LIDAR_Te.mat",
        "sha256": [
            "d639fe5d300390e093a43e5362ff760c4991890a51d8dbd7fda4fd6d6b104b76",
            "efc8480b0a6d28c56ce5afca3fa896308c762b9b1982d5df0e524473f68d0f68",
            "71ecf2dc1b97e7b5b8ad0b507c45293c3c4feb78a6958de7418fc57f71ef8fb5"
        ]
    },
    "Trento": {
        "display_name": "Trento",
        "hsi_channels": 63,
        "aux_channels": 1,
        "classes": 6,
        "sdh_rank": 16,
        "parameters": 652591,
        "aux_file": "LIDAR_Te.mat",
        "sha256": [
            "d81e95b3f7472a86cf2124275621aa8f62b5e6888c14ca77c99ae311a883e063",
            "31d05c73d0935bf82ffd0a85eb4840136634f0729ec2cdb412e1f3928cb7a8b9",
            "6d76b092aff3e709e6d18c97568c4f7cf438256a1d534bc4574373e3e1f5429f"
        ]
    },
    "MUUFL": {
        "display_name": "MUUFL",
        "hsi_channels": 64,
        "aux_channels": 2,
        "classes": 11,
        "sdh_rank": 16,
        "parameters": 639451,
        "aux_file": "LIDAR_Te.mat",
        "sha256": [
            "6054b750cac8786c42a040b6f843cf60dd2b4023c47a50854947cc9eeee97b69",
            "d0cec97d9c710fcce6160be6f1387a13881ecae7b03b3f097c38fe7f33314899",
            "fbda39143d707443db9ccdd9f5f32f034d9c8dceb55ee433e51ec9f27cc8f5f0"
        ]
    },
    "Houston2018": {
        "display_name": "Houston2018",
        "hsi_channels": 50,
        "aux_channels": 1,
        "classes": 20,
        "sdh_rank": 16,
        "parameters": 629521,
        "aux_file": "LIDAR_Te.mat",
        "sha256": [
            "1d340f4b4b2de8029634c49675afda1480a73ba691b7ea9b9f3d241fa8f49244",
            "b33724f8ff51894fa9a9d654ff7daaf1896d2aaee2e6741e4209332f19d0e8d0",
            "2209788de38cf3977ffae9d6f69d927389a592c4c6442798178fd5dd2374621e"
        ]
    },
    "Augsburg": {
        "display_name": "Augsburg",
        "hsi_channels": 180,
        "aux_channels": 1,
        "classes": 7,
        "sdh_rank": 16,
        "parameters": 734587,
        "aux_file": "LIDAR_Te.mat",
        "sha256": [
            "94d972819e2a34c3d6a0a793fafe5adf52c9e006a1bb5b7e194763981ea1218e",
            "229c100c3d4edfa3b5a0d4ef06f1519fd36ee15a00a7647e6e1974bb1cb754ee",
            "35bf32da2ee180c6e8012e9386e49ec37cb216e12b9e305acb7fccc183b562c9"
        ]
    }
}


def to_tensor(array, channels):
    if array.ndim != 4 or array.shape[1:3] != (11, 11) or array.shape[3] != channels:
        raise ValueError(f"Expected [N,11,11,{channels}], got {array.shape}")
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).permute(0, 3, 1, 2).reshape(
        len(array), channels, 121
    )


def matlab_batches(path, channels, batch_size):
    """Yield consecutive batches without loading a MATLAB v7.3 cube into RAM."""
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            data = handle["Data"]
            if data.shape[:3] != (channels, 11, 11):
                raise ValueError(f"Unexpected v7.3 Data shape {data.shape} in {path}")
            for start in range(0, data.shape[3], batch_size):
                stop = min(start + batch_size, data.shape[3])
                physical = data.astype(np.float32)[..., start:stop]
                batch = np.ascontiguousarray(physical.transpose(3, 0, 2, 1))
                yield start, torch.from_numpy(batch.reshape(stop - start, channels, 121))
    else:
        data = scipy.io.loadmat(path, variable_names=["Data"])["Data"]
        if data.ndim != 4 or data.shape[1:3] != (11, 11) or data.shape[3] != channels:
            raise ValueError(f"Unexpected Data shape {data.shape} in {path}")
        for start in range(0, len(data), batch_size):
            yield start, to_tensor(data[start:start + batch_size], channels)


def hsi_paths(directory, dataset):
    if dataset != "Augsburg":
        return [directory / "HSI_Te.mat"]
    paths = sorted(
        directory.glob("HSI_Te_Part*.mat"),
        key=lambda path: int(path.stem.removeprefix("HSI_Te_Part")),
    )
    numbers = [int(path.stem.removeprefix("HSI_Te_Part")) for path in paths]
    if not numbers or numbers != list(range(1, len(paths) + 1)):
        raise ValueError(f"Missing or nonconsecutive Augsburg HSI parts: {numbers}")
    return paths


def hsi_batches(directory, dataset, channels, batch_size):
    offset = 0
    for path in hsi_paths(directory, dataset):
        for start, batch in matlab_batches(path, channels, batch_size):
            yield offset + start, batch
        offset += mat_samples(path)


def mat_samples(path):
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as handle:
            return int(handle["Data"].shape[3])
    variables = scipy.io.whosmat(path)
    shape = next(shape for name, shape, _ in variables if name == "Data")
    return int(shape[0])


def checkpoint_path(dataset, rank, spec):
    if rank not in (1, 2, 3):
        raise ValueError(f"Invalid checkpoint rank: {rank}")
    return ROOT / "checkpoints" / dataset / f"{spec['display_name']}_{rank}.pkl"


def verify_hash(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {actual} != {expected}")


def build_model(spec):
    model = MFT_ConformerSDHSoftAnchor(
        FM=16, NC=spec["hsi_channels"], NCLidarMS=spec["aux_channels"],
        Classes=spec["classes"], detach_lidar=True, use_dilation=True,
        use_aux_loss=True, gate_last_bias=-2.0, conformer_depth=2,
        conformer_heads=4, conformer_ffn_dim=256, conformer_conv_kernel=15,
        sdh_rank=spec["sdh_rank"], sdh_alpha_init=0.1, anchor_temp=2.0,
    )
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != spec["parameters"]:
        raise ValueError(f"Parameter count {count} != {spec['parameters']}")
    return model


def load_model(dataset, spec, rank, device):
    path = checkpoint_path(dataset, rank, spec)
    verify_hash(path, spec["sha256"][rank - 1])
    model = build_model(spec)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def metrics(labels, predictions, classes):
    matrix = np.bincount(
        classes * labels + predictions, minlength=classes * classes
    ).reshape(classes, classes)
    support = matrix.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Every class must occur in the evaluated labels")
    total = int(matrix.sum())
    observed = float(np.trace(matrix)) / total
    expected = float(np.dot(support.astype(float), matrix.sum(axis=0))) / total ** 2
    return {
        "oa": observed * 100,
        "aa": float(np.mean(np.diag(matrix) / support)) * 100,
        "kappa": (observed - expected) / (1 - expected) * 100,
    }


def evaluate(dataset, spec, data_root, batch_size, device, max_samples, output, report):
    directory = data_root / dataset
    paths = hsi_paths(directory, dataset)
    full_count = sum(mat_samples(path) for path in paths)
    sample_count = full_count
    if max_samples is not None:
        sample_count = min(sample_count, max_samples)
    aux_path = directory / spec["aux_file"]
    aux = to_tensor(scipy.io.loadmat(aux_path, variable_names=["Data"])["Data"],
                    spec["aux_channels"])
    if len(aux) != full_count:
        raise ValueError(f"Auxiliary data has {len(aux)} samples, HSI has {full_count}")
    label_path = directory / "TeLabel.mat"
    labels = None
    if label_path.exists():
        labels = scipy.io.loadmat(label_path, variable_names=["Data"])["Data"].reshape(-1)
        labels = labels.astype(np.int64) - 1
        if len(labels) != full_count or np.any((labels < 0) | (labels >= spec["classes"])):
            raise ValueError("Invalid test label count or 1-based class IDs")
        labels = labels[:sample_count]
    results = {}
    saved = {}
    for rank in (1, 2, 3):
        model = load_model(dataset, spec, rank, device)
        predictions = np.empty(sample_count, dtype=np.int64)
        with torch.inference_mode():
            for start, hsi in hsi_batches(directory, dataset, spec["hsi_channels"], batch_size):
                if start >= sample_count:
                    break
                stop = min(start + len(hsi), sample_count)
                logits = model(hsi[:stop - start].to(device), aux[start:stop].to(device))
                if isinstance(logits, tuple):
                    logits = logits[0]
                predictions[start:stop] = logits.argmax(dim=1).cpu().numpy()
        saved[f"run{rank}"] = predictions
        if labels is not None and max_samples is None:
            row = metrics(labels, predictions, spec["classes"])
            results[rank] = row
            print(f"{dataset} run{rank}: OA={row['oa']:.4f} "
                  f"AA={row['aa']:.4f} kappa*100={row['kappa']:.4f}", flush=True)
        else:
            print(f"{dataset} run{rank}: {len(predictions)} predictions", flush=True)
        del model
    summary = {}
    if results:
        for name in ("oa", "aa", "kappa"):
            values = np.array([results[rank][name] for rank in (1, 2, 3)])
            summary[name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}
            print(f"{dataset} three-run {name}: {values.mean():.4f} +/- "
                  f"{values.std(ddof=0):.4f}", flush=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **saved)
        print(f"Saved 0-based predictions to {output}", flush=True)
    if report is not None:
        record = {
            "dataset": dataset,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root.resolve()),
            "test_samples": sample_count,
            "device": str(device),
            "runs": [1, 2, 3],
            "checkpoint_sha256": spec["sha256"],
            "run_metrics": {f"run{rank}": results[rank] for rank in results},
            "three_run_summary": summary,
            "full_test_set": max_samples is None,
        }
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Saved verification report to {report}", flush=True)


def verify_all(datasets, device):
    expected = {
        checkpoint_path(dataset, rank, spec).resolve()
        for dataset, spec in datasets.items()
        for rank in (1, 2, 3)
    }
    found = {path.resolve() for path in (ROOT / "checkpoints").glob("*/*.pkl")}
    if len(expected) != 15 or expected != found:
        raise ValueError(f"Expected exactly 15 selected weights; missing={expected - found}, extra={found - expected}")
    count = 0
    for dataset, spec in datasets.items():
        for rank in (1, 2, 3):
            model = load_model(dataset, spec, rank, device)
            with torch.inference_mode():
                hsi = torch.zeros(1, spec["hsi_channels"], 121, device=device)
                aux = torch.zeros(1, spec["aux_channels"], 121, device=device)
                logits = model(hsi, aux)
            if logits.shape != (1, spec["classes"]):
                raise ValueError(f"Wrong output shape for {dataset} run{rank}")
            print(f"OK {dataset} run{rank}", flush=True)
            count += 1
            del model
    print(f"Verified {count} checkpoints with SHA-256, strict load and forward pass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--data-root", type=Path, default=ROOT.parents[1],
                        help="Directory containing Houston, Trento, MUUFL, Houston2018 and Augsburg")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, help="Prefix-only smoke test; no paper metrics")
    parser.add_argument("--output", type=Path, help="Optional .npz of 0-based predictions")
    parser.add_argument("--report", type=Path, help="Write per-run metrics to a JSON file")
    parser.add_argument("--verify-all", action="store_true",
                        help="Check all 15 checkpoint hashes, strict loads and forward passes")
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.max_samples is not None and args.max_samples <= 0):
        parser.error("Batch size and max samples must be positive")
    if not args.verify_all and args.dataset is None:
        parser.error("Specify --dataset or --verify-all")
    device = torch.device(args.device)
    if args.verify_all:
        verify_all(DATASETS, device)
    if args.dataset:
        evaluate(args.dataset, DATASETS[args.dataset], args.data_root,
                 args.batch_size, device, args.max_samples, args.output, args.report)


if __name__ == "__main__":
    main()
