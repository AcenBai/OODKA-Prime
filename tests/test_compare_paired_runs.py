import csv
import json

import pytest

from scripts.compare_paired_segmentation_runs import (
    compare_paired_runs,
    main,
    read_metrics,
)


def _write_metrics(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "dice_mean_gt", "dice_1", "dice_2"]
        )
        writer.writeheader()
        writer.writerows(rows)


def _row(case_id, overall, dice_1=None, dice_2=None):
    return {
        "case_id": case_id,
        "dice_mean_gt": overall,
        "dice_1": overall if dice_1 is None else dice_1,
        "dice_2": overall if dice_2 is None else dice_2,
    }


def test_paired_join_uses_case_id_and_reports_wins_ties_losses(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_metrics(
        baseline,
        [
            _row("case_a", 0.5, 0.4, 0.6),
            _row("case_b", 0.6, 0.5, 0.7),
            _row("case_c", 0.7, 0.6, 0.8),
        ],
    )
    # Deliberately reorder rows: deltas by case are +0.1, 0.0, -0.1.
    _write_metrics(
        candidate,
        [
            _row("case_c", 0.6, 0.65, 0.85),
            _row("case_a", 0.6, 0.45, 0.65),
            _row("case_b", 0.6, 0.55, 0.75),
        ],
    )

    summary, rows = compare_paired_runs(
        baseline, [("roi_pad", candidate)], bootstrap_samples=1_000, seed=17
    )

    assert summary["case_count"] == 3
    assert summary["baseline"]["overall_mean"] == pytest.approx(0.6)
    assert summary["runs"]["roi_pad"]["overall_mean"] == pytest.approx(0.6)
    assert summary["runs"]["roi_pad"]["per_class_mean"] == pytest.approx(
        {"dice_1": 0.55, "dice_2": 0.75}
    )
    comparison = summary["runs"]["roi_pad"]["comparison_to_baseline"]
    assert comparison["paired_mean_delta"] == pytest.approx(0.0, abs=1e-15)
    assert (comparison["wins"], comparison["ties"], comparison["losses"]) == (1, 1, 1)
    assert [row["case_id"] for row in rows] == ["case_a", "case_b", "case_c"]
    assert rows[0]["roi_pad__delta_vs_baseline"] == pytest.approx(0.1)
    assert rows[2]["roi_pad__delta_vs_baseline"] == pytest.approx(-0.1)


def test_constant_paired_delta_has_degenerate_reproducible_ci(tmp_path):
    baseline = tmp_path / "baseline.csv"
    run_a = tmp_path / "run_a.csv"
    run_b = tmp_path / "run_b.csv"
    _write_metrics(baseline, [_row("a", 0.2), _row("b", 0.4), _row("c", 0.6)])
    _write_metrics(run_a, [_row("a", 0.25), _row("b", 0.45), _row("c", 0.65)])
    _write_metrics(run_b, [_row("a", 0.25), _row("b", 0.45), _row("c", 0.65)])

    summary, _ = compare_paired_runs(
        baseline,
        [("a", run_a), ("b", run_b)],
        bootstrap_samples=321,
        seed=123,
    )
    comparison_a = summary["runs"]["a"]["comparison_to_baseline"]
    comparison_b = summary["runs"]["b"]["comparison_to_baseline"]
    assert comparison_a["paired_mean_delta"] == pytest.approx(0.05)
    assert comparison_a["bootstrap_95_ci"] == pytest.approx([0.05, 0.05])
    # Each run receives the same fixed-seed case resamples.
    assert comparison_a["bootstrap_95_ci"] == comparison_b["bootstrap_95_ci"]


def test_strict_join_rejects_missing_or_unexpected_case_ids(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    _write_metrics(baseline, [_row("a", 0.2), _row("b", 0.4)])
    _write_metrics(candidate, [_row("a", 0.3), _row("c", 0.5)])

    with pytest.raises(ValueError, match=r"missing=\['b'\], unexpected=\['c'\]"):
        compare_paired_runs(baseline, [("candidate", candidate)])


def test_reader_rejects_duplicate_case_ids(tmp_path):
    metrics = tmp_path / "duplicate.csv"
    _write_metrics(metrics, [_row("a", 0.2), _row("a", 0.3)])

    with pytest.raises(ValueError, match="Duplicate case_id='a'"):
        read_metrics(metrics)


def test_reader_allows_missing_per_class_but_not_missing_overall(tmp_path):
    metrics = tmp_path / "missing_class.csv"
    _write_metrics(
        metrics,
        [
            _row("a", 0.2, dice_1="", dice_2=0.3),
            _row("b", 0.4, dice_1=0.5, dice_2=""),
        ],
    )
    table = read_metrics(metrics)
    assert table.values["dice_1"]["a"] != table.values["dice_1"]["a"]

    missing_overall = tmp_path / "missing_overall.csv"
    _write_metrics(missing_overall, [_row("a", "")])
    with pytest.raises(ValueError, match="Invalid numeric value"):
        read_metrics(missing_overall)


def test_cli_writes_json_and_per_case_csv(tmp_path):
    baseline = tmp_path / "baseline.csv"
    candidate = tmp_path / "candidate.csv"
    output_dir = tmp_path / "comparison"
    _write_metrics(baseline, [_row("a", 0.2), _row("b", 0.4)])
    _write_metrics(candidate, [_row("b", 0.5), _row("a", 0.3)])

    main(
        [
            "--baseline",
            str(baseline),
            "--run",
            f"oracle_pad={candidate}",
            "--output-dir",
            str(output_dir),
            "--bootstrap-samples",
            "50",
            "--seed",
            "9",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runs"]["oracle_pad"]["comparison_to_baseline"][
        "paired_mean_delta"
    ] == pytest.approx(0.1)
    with (output_dir / "per_case_deltas.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == [
        "case_id",
        "baseline__dice_mean_gt",
        "oracle_pad__dice_mean_gt",
        "oracle_pad__delta_vs_baseline",
    ]
    assert [row["case_id"] for row in rows] == ["a", "b"]
