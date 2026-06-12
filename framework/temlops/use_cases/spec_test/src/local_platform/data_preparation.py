import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

import great_expectations as gx
import pandas as pd
from temlops.src.artifact_types import Configuration, Data


FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DATA_PATH = os.path.join(FOLDER_PATH, "artifacts", "data")
ARTIFACTS_REPORT_PATH = os.path.join(FOLDER_PATH, "artifacts", "report")


def _as_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _resolve_path(filepath: str) -> str:
    if os.path.isabs(filepath):
        return filepath
    return os.path.join(FOLDER_PATH, filepath)


def _ensure_parent_dir(filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def _load_dataset_with_fallback(data: Data) -> pd.DataFrame:
    try:
        dataset = data.load_dataset()
        if isinstance(dataset, pd.DataFrame):
            return dataset
        return pd.DataFrame(dataset)
    except FileNotFoundError:
        resolved_path = _resolve_path(data.filepath)
        if resolved_path.endswith(".csv"):
            return pd.read_csv(resolved_path)
        if resolved_path.endswith(".json"):
            return pd.read_json(resolved_path)
        return pd.read_parquet(resolved_path)


def data_profiling_quantity_ge(
    product_name: str,
    data: Data,
    config: Configuration,
) -> Dict[str, Any]:
    """Validate dataset quantity using Great Expectations row-count bounds."""
    dataset = _load_dataset_with_fallback(data)

    min_rows = int(getattr(config, "min_rows", 0))
    max_rows = _as_int_or_none(getattr(config, "max_rows", None))

    ge_validator = gx.from_pandas(dataset)
    ge_result = ge_validator.expect_table_row_count_to_be_between(
        min_value=min_rows,
        max_value=max_rows,
    )

    default_report_path = os.path.join(
        ARTIFACTS_REPORT_PATH,
        "data_quantity_profile_report.json",
    )
    report_filepath = _resolve_path(
        getattr(config, "report_filepath", default_report_path)
    )
    _ensure_parent_dir(report_filepath)

    profile_report = {
        "product_name": product_name,
        "operation": "data_profiling_quantity_ge",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset_path": data.filepath,
        "dataset_path_resolved": _resolve_path(data.filepath),
        "row_count": int(len(dataset)),
        "min_rows": min_rows,
        "max_rows": max_rows,
        "success": bool(ge_result.success),
        "expectation_type": ge_result.expectation_config.expectation_type,
        "result": ge_result.result,
    }

    with open(report_filepath, "w", encoding="utf-8") as report_file:
        json.dump(profile_report, report_file, indent=2)

    return profile_report


def data_validation_minimum_columns(
    product_name: str,
    data: Data,
    config: Configuration,
) -> Dict[str, Any]:
    """Validate that required columns exist in the dataset."""
    dataset = _load_dataset_with_fallback(data)
    required_columns = getattr(config, "required_columns", [])
    missing_columns = [column for column in required_columns if column not in dataset.columns]

    return {
        "product_name": product_name,
        "operation": "data_validation_minimum_columns",
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "success": len(missing_columns) == 0,
    }


def data_preprocessing_drop_duplicates(
    product_name: str,
    data: Data,
    config: Configuration,
) -> Dict[str, Any]:
    """Apply minimal preprocessing by removing duplicate rows."""
    dataset = _load_dataset_with_fallback(data)
    deduplicated = dataset.drop_duplicates()

    output_filepath = _resolve_path(
        getattr(
        config,
        "output_filepath",
        os.path.join(ARTIFACTS_DATA_PATH, "spec_test_preprocessed.json"),
        )
    )
    _ensure_parent_dir(output_filepath)

    output_data = Data(
        filepath=output_filepath,
        platform=data.platform,
        product_name=product_name,
    )
    output_data.log_dataset(deduplicated)

    return {
        "product_name": product_name,
        "operation": "data_preprocessing_drop_duplicates",
        "input_rows": int(len(dataset)),
        "output_rows": int(len(deduplicated)),
        "output_filepath": output_filepath,
    }


def data_documentation_generate_summary(
    product_name: str,
    data: Data,
    config: Configuration,
) -> Dict[str, Any]:
    """Generate markdown documentation from profiling-ready dataset metadata."""
    dataset = _load_dataset_with_fallback(data)
    documentation_path = _resolve_path(
        getattr(
            config,
            "documentation_filepath",
            os.path.join("..", "..", "docs", "Datasheet.md"),
        )
    )

    _ensure_parent_dir(documentation_path)

    summary_lines = [
        f"# {product_name} - Data Summary",
        "",
        f"- Rows: {len(dataset)}",
        f"- Columns: {len(dataset.columns)}",
        f"- Column names: {', '.join(map(str, dataset.columns.tolist()))}",
    ]

    with open(documentation_path, "w", encoding="utf-8") as docs_file:
        docs_file.write("\n".join(summary_lines) + "\n")

    return {
        "product_name": product_name,
        "operation": "data_documentation_generate_summary",
        "documentation_filepath": documentation_path,
    }


# Sources: [1] https://docs.greatexpectations.io/docs/core/introduction/try_gx/ , [2] framework/library/src/artifact_types.py
