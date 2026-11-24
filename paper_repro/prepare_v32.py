"""Materialize a minimal five-type port of the pinned DHGCMDA release."""

import difflib
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_release import HERE, RELEASE, REVISION, ROOT

DATA = HERE / "data"
PORT = HERE / "v32_release"
DATASET = "v3.2_spld_paper"
SHAPE = (411, 271, 5)
SOURCE_TYPES = ("genetics", "epigenetics", "circulation", "target", "tissue")
RELEASE_TYPES = ("circulation", "epigenetics", "target", "genetics", "tissue")
ASSOCIATIONS = "multi_all_mirna_disease_pairs_without_negative.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def gip(association):
    degree = association.sum(axis=1)
    distances = degree[:, None] + degree[None, :] - 2 * association @ association.T
    return np.exp(-len(degree) * distances / degree.sum())


def materialize_data():
    source = ROOT / DATASET
    inputs = ROOT / "HMDD_data" / "MDAv3.2-3"
    rows = np.loadtxt(source / "triplets.csv", delimiter=",", skiprows=1, dtype=np.int64)
    source_tensor = np.load(source / "Y_multilabel.npy", allow_pickle=False)
    if (source_tensor.shape, rows.shape) != (SHAPE, (11748, 3)):
        raise ValueError("Unexpected SPLD dimensions")
    reconstructed = np.zeros(SHAPE, dtype=bool)
    reconstructed[tuple(rows.T)] = True
    if len(np.unique(rows, axis=0)) != len(rows) or not np.array_equal(reconstructed, source_tensor):
        raise ValueError("Triplets do not reproduce the published SPLD tensor")
    for index, name in enumerate(SOURCE_TYPES):
        raw = np.loadtxt(inputs / f"{name}.csv", delimiter=",", skiprows=1, usecols=range(1, 272))
        if not np.array_equal(raw, source_tensor[..., index]):
            raise ValueError(f"SPLD channel mapping differs for {name}")

    mapping = np.array([RELEASE_TYPES.index(name) + 1 for name in SOURCE_TYPES])
    release_rows = rows + 1
    release_rows[:, 2] = mapping[rows[:, 2]]
    release_rows = release_rows[np.argsort(release_rows[:, 2], kind="stable")]
    binary = reconstructed.any(axis=-1).astype(np.float64)
    if int(binary.sum()) != 8735:
        raise ValueError("Unexpected number of positive pairs")
    collapsed = np.zeros(SHAPE[:2], dtype=int)
    np.maximum.at(collapsed, (release_rows[:, 0] - 1, release_rows[:, 1] - 1), release_rows[:, 2])
    counts = np.bincount(collapsed.ravel(), minlength=6)[1:].tolist()

    misim = np.load(source / "M_MISIM.npy", allow_pickle=False)
    mesh = np.load(source / "D_MESH.npy", allow_pickle=False)
    if (misim.shape, mesh.shape) != ((411, 411), (271, 271)):
        raise ValueError("Unexpected similarity shapes")

    variants = {}
    for variant, disease_view in (("standard", mesh), ("diagnostic-disease-gip", gip(binary.T))):
        directory = DATA / variant / DATASET
        directory.mkdir(parents=True, exist_ok=True)
        files = {"M_FSM.txt": misim, "M_GSM.txt": gip(binary),
                 "D_SSM1.txt": mesh, "D_SSM2.txt": disease_view}
        for name, matrix in files.items():
            np.savetxt(directory / name, matrix, fmt="%.17g")
            if not np.array_equal(np.loadtxt(directory / name), matrix):
                raise RuntimeError(f"Similarity text changed values: {name}")
        np.savetxt(directory / ASSOCIATIONS, release_rows, delimiter=",", fmt="%d")
        variants[variant] = {item.name: sha(item) for item in sorted(directory.iterdir())}
    return {"shape": list(SHAPE), "triplets": len(rows), "positive_pairs": int(binary.sum()),
            "positive_pair_density": float(binary.mean()), "release_types": list(RELEASE_TYPES),
            "collapsed_counts": counts, "mirna_gamma": len(binary) / float(binary.sum()),
            "disease_second_view": "D_MESH duplicated for standard, GIP reserved for G3 diagnostic",
            "source_sha256": {name: sha(source / name) for name in
                              ("triplets.csv", "Y_multilabel.npy", "M_MISIM.npy", "D_MESH.npy")},
            "variants": variants}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r} in pinned release")
    return source.replace(old, new, 1)


def materialize_port(counts):
    manifest = json.loads((HERE / "release_manifest.json").read_text(encoding="utf-8"))
    if manifest["commit"] != REVISION:
        raise ValueError("Unexpected upstream revision")
    PORT.mkdir(exist_ok=True)
    patches = {
        "prepareData.py": (
            ("max_mirna = 495", "max_mirna = 411"),
            ("max_disease = 383", "max_disease = 271"),
            ("'type4': [],", "'type4': [],\n            'type5': [],"),
            ("type_indices['type4'].append([i, j])",
             "type_indices['type4'].append([i, j])\n            elif value == 5:\n                type_indices['type5'].append([i, j])"),
            ("'type4': type_indices['type4'],", "'type4': type_indices['type4'],\n                'type5': type_indices['type5'],"),
            ('4: "遗传学类型 (Genetics)"', '4: "遗传学类型 (Genetics)",\n            5: "Tissue"'),
        ),
        "param.py": (
            ("if args.num_association_types != 4:", "if args.num_association_types != 5:"),
            ("expects 4 types: circulation, epigenetics, target, genetics.",
             "expects 5 types: circulation, epigenetics, target, genetics, tissue."),
            ("args.mi_num = 495", "args.mi_num = 411"),
            ("args.dis_num = 383", "args.dis_num = 271"),
            ("~1498", "8735"),
            ("Types: 4 (Circulation, Epigenetics, Target, Genetics)",
             "Types: 5 (Circulation, Epigenetics, Target, Genetics, Tissue)"),
            ("counts = [367, 157, 293, 681]", f"counts = {counts}"),
        ),
        "hetero_model.py": (
            ("num_types=4\n        )", "num_types=args.num_association_types\n        )"),
            ("torch.tensor([0.219, 0.094, 0.175, 0.407])",
             f"torch.tensor({counts}, dtype=torch.float32) / {sum(counts)}"),
        ),
        "main_experiments_hetero1.py": (
            ("counts = [367, 157, 293, 681]", f"counts = {counts}"),
            ("        type_indices[pos_targets == 4] = 3",
             "        type_indices[pos_targets == 4] = 3\n        type_indices[pos_targets == 5] = 4"),
            ("torch.zeros((495, 380), device=device)", "torch.zeros((411, 271), device=device)"),
        ),
    }
    changes = []
    files = {}
    for entry in manifest["files"]:
        path = Path(entry["path"])
        if path.parent != Path(".") or path.suffix != ".py":
            continue
        original = RELEASE / path
        if sha(original) != entry["sha256"]:
            raise ValueError(f"Pinned file changed: {path}")
        target = PORT / path
        if path.name in patches:
            before = original.read_text(encoding="utf-8")
            after = before
            for old, new in patches[path.name]:
                after = replace_once(after, old, new)
            if path.name in {"param.py", "prepareData.py"}:
                after = after.replace("v2.0_495m383D", DATASET)
            if path.name == "prepareData.py":
                after = after.replace("495 miRNAs × 383 Diseases", "411 miRNAs × 271 Diseases")
            changes.extend(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                                fromfile=str(path), tofile=f"v32/{path}", lineterm=""))
            target.write_text(after, encoding="utf-8")
        else:
            target.write_bytes(original.read_bytes())
        files[path.as_posix()] = sha(target)

    (HERE / "PORT_DIFF.md").write_text(
        "# Port HMDD v3.2\n\n"
        "Mã nền `06f9ba3`. Chỉ thay kích thước dữ liệu, số loại, ánh xạ nhãn "
        "và số đếm lớp. Trình chấm điểm giữ nguyên từng byte. Top-1 cho 5 loại "
        "được ghi riêng từ dự đoán đã lưu. Bản phát hành gốc nằm trong `release/`.\n\n"
        + "```diff\n" + "\n".join(changes) + "\n```\n", encoding="utf-8")
    return files


def main():
    data = materialize_data()
    code = materialize_port(data["collapsed_counts"])
    save_json(DATA / "manifest.json", {"release": REVISION, "data": data, "code_sha256": code})
    print(json.dumps(data, indent=2), flush=True)


if __name__ == "__main__":
    main()
