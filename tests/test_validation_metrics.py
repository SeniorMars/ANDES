import json
from typing import cast

import numpy as np
import pandas as pd
import pytest

from experiments.validate_drug_disease_matrix import (
    _average_precision,
    _comparison_metrics,
    _validation_report,
    main,
)


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([1, 0, 1], [3.0, 2.0, 1.0], 5.0 / 6.0),
        ([1, 0], [2.0, 2.0], 0.5),
        ([1, 1, 1], [0.0, -1.0, 2.0], 1.0),
    ],
)
def test_average_precision_exact_cases(labels, scores, expected):
    assert _average_precision(labels, scores) == pytest.approx(expected)


def test_average_precision_returns_nan_without_positives():
    assert np.isnan(_average_precision([0, 0], [2.0, 1.0]))


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([0, 2], [2.0, 1.0], "binary"),
        ([0, 1], [np.inf, 1.0], "finite"),
        ([0, 1], [np.nan, 1.0], "finite"),
        ([0], [1.0, 2.0], "same-length"),
    ],
)
def test_average_precision_rejects_invalid_inputs(labels, scores, message):
    with pytest.raises(ValueError, match=message):
        _average_precision(labels, scores)


def test_validation_report_applies_coverage_and_agreement_envelope():
    reference = pd.DataFrame(
        [[3.0, 2.0], [1.0, 0.0]],
        index=["drug-a", "drug-b"],
        columns=["disease-a", "disease-b"],
    )
    matching = _validation_report(
        reference,
        reference.copy(),
        requested_drugs=2,
        requested_diseases=2,
    )
    assert matching["passed"] is True
    assert matching["failures"] == []

    reversed_scores = pd.DataFrame(
        [[0.0, 1.0], [2.0, 3.0]],
        index=reference.index,
        columns=reference.columns,
    )
    failed = _validation_report(
        reference.iloc[:1],
        reversed_scores.iloc[:1],
        requested_drugs=2,
        requested_diseases=2,
    )
    assert failed["passed"] is False
    assert failed["drug_coverage"] == 0.5
    failures = cast(list[str], failed["failures"])
    assert any("drug coverage" in failure for failure in failures)
    assert any("correlation" in failure for failure in failures)


def test_validation_rejects_candidate_nonfinite_values_before_masking():
    reference = pd.DataFrame([[1.0, 2.0], [3.0, np.nan]])
    candidate = pd.DataFrame([[1.0, np.inf], [3.0, np.nan]])

    with pytest.raises(ValueError, match="1 non-finite"):
        _comparison_metrics(reference, candidate)

    report = _validation_report(
        reference,
        candidate,
        requested_drugs=2,
        requested_diseases=2,
    )
    assert report["passed"] is False
    assert report["reference_finite_values"] == 3
    assert report["jointly_finite_values"] == 2
    assert report["jointly_finite_fraction"] == pytest.approx(2.0 / 3.0)


def test_matrix_validation_failure_status_and_report_only_mode(tmp_path):
    embedding = tmp_path / "embedding.csv"
    genes = tmp_path / "genes.txt"
    drugs = tmp_path / "drugs.gmt"
    diseases = tmp_path / "diseases.gmt"
    reference = tmp_path / "reference.csv"
    np.savetxt(
        embedding,
        np.eye(4, dtype=np.float32),
        delimiter=",",
    )
    genes.write_text("g0\ng1\ng2\ng3\n", encoding="utf-8")
    drugs.write_text(
        "drug-a\tdescription\tg0\tg1\ndrug-b\tdescription\tg2\tg3\n",
        encoding="utf-8",
    )
    diseases.write_text(
        "disease-a\tdescription\tg0\tg2\ndisease-b\tdescription\tg1\tg3\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [[3.0, 2.0], [1.0, 0.0]],
        index=["drug-a", "drug-b"],
        columns=["disease-a", "disease-b"],
    ).to_csv(reference)

    common_args = [
        "--emb",
        str(embedding),
        "--genelist",
        str(genes),
        "--drug-gmt",
        str(drugs),
        "--disease-gmt",
        str(diseases),
        "--reference-zscores",
        str(reference),
        "--min-size",
        "1",
        "--max-size",
        "4",
        "--ite",
        "2",
    ]
    failed_output = tmp_path / "failed"
    assert main([*common_args, "--out-dir", str(failed_output)]) == 1
    report = json.loads(
        (failed_output / "drug_disease_validation.json").read_text(encoding="utf-8")
    )
    assert report["passed"] is False
    assert report["failures"]

    report_only_output = tmp_path / "report-only"
    assert (
        main(
            [
                *common_args,
                "--out-dir",
                str(report_only_output),
                "--report-only",
            ]
        )
        == 0
    )
    report_only = json.loads(
        (report_only_output / "drug_disease_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert report_only["passed"] is False
