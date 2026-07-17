"""Dataset-level manifest loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from apple_pi.constants import BENCHMARK_NAME, NUM_ROLLOUTS


@dataclass(frozen=True)
class DatasetCase:
    case_id: str
    path: str
    split: str = "test"


@dataclass(frozen=True)
class DatasetIndex:
    root: Path
    version: str
    cases: tuple[DatasetCase, ...]
    num_rollouts: int = NUM_ROLLOUTS


def load_dataset_index(root: str | Path) -> DatasetIndex:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "dataset.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Download the GT repository or follow "
            "docs/GT_FORMAT.md to prepare it."
        )
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if data.get("name") != BENCHMARK_NAME:
        raise ValueError(
            f"Expected dataset name {BENCHMARK_NAME!r}, got {data.get('name')!r}"
        )
    cases = tuple(
        DatasetCase(
            case_id=str(item["case_id"]),
            path=str(item["path"]),
            split=str(item.get("split", "test")),
        )
        for item in data.get("cases", [])
    )
    if not cases:
        raise ValueError(f"No cases listed in {manifest_path}")
    num_rollouts = int(data.get("num_rollouts", NUM_ROLLOUTS))
    if num_rollouts != NUM_ROLLOUTS:
        raise ValueError(
            f"Apple-PI release protocol requires {NUM_ROLLOUTS} rollouts; "
            f"dataset declares {num_rollouts}"
        )
    return DatasetIndex(
        root=root,
        version=str(data["version"]),
        cases=cases,
        num_rollouts=num_rollouts,
    )
