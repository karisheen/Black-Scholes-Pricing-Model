"""Real synthetic-run regression tests; no canned measured experiment output."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_pricing.hedge_experiment import (
    SCHEMA_VERSION,
    _arrays,
    _comparisons,
    _digest,
    _source_fingerprint,
    _summarize,
    load_config,
    main,
    parse_config,
    run_experiment,
)
from quant_pricing.hedge_paths import GBMConfig, generate_gbm
from quant_pricing.hedging import (
    HedgeContract,
    HedgeCosts,
    HedgePolicy,
    prepare_market,
    simulate_hedge,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hedging_experiment.json"


@pytest.fixture
def document() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["market"].update(steps=8, paths=12, seeds=[1729, 2718])
    data["policies"][3]["every"] = 8
    return data


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    data = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    data["market"].update(steps=8, paths=12, seeds=[1729, 2718])
    data["policies"][3]["every"] = 8
    directory = tmp_path_factory.mktemp("hedge_measurements")
    return directory, run_experiment(parse_config(data), directory)


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_example_is_predeclared_bounded_grid() -> None:
    config = load_config(EXAMPLE)
    assert config.schema_version == SCHEMA_VERSION
    assert config.market.seeds == (1729, 2718, 31415)
    assert config.market.paths == 512
    assert config.market.steps == 64
    assert config.market.path_volatilities == (0.12, 0.2, 0.32)
    assert len(config.policies) == 6
    assert len(config.cost_schedules) == 3
    assert config.market.drift == config.contract.rate - config.contract.dividend_yield
    assert config.contract.dividend_yield == 0
    assert parse_config(config.to_dict()) == config


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "schema_version", "future", "schema_version"),
        (None, "extra", 1, "unknown fields"),
        (None, "experiment_id", "<script>", "ASCII name"),
        (None, "experiment_id", "", "ASCII name"),
        (None, "zero_cost_reference", "missing", "configured cost schedule"),
        (None, "zero_cost_reference", "baseline", "exactly zero"),
        (None, "policies", [], "nonempty array"),
        (None, "cost_schedules", "zero", "nonempty array"),
        ("contract", "kind", "binary", "call or put"),
        ("contract", "strike", True, "finite number"),
        ("contract", "rate", "0.03", "finite number"),
        ("contract", "rate", float("inf"), "finite number"),
        ("contract", "rate", 10**1000, "finite number"),
        ("contract", "strike", -1, "strike"),
        ("market", "steps", False, "integer"),
        ("market", "steps", 1.5, "integer"),
        ("market", "steps", 0, "integer"),
        ("market", "paths", 1, "integer"),
        ("market", "seeds", [], "nonempty array"),
        ("market", "seeds", [1, 1], "unique"),
        ("market", "path_volatilities", [0.2, 0.2], "unique"),
        ("market", "path_volatilities", [-0.1], "volatility"),
        ("example", "seed", 999, "configured scenario"),
        ("example", "path_volatility", 0.9, "configured scenario"),
        ("example", "policy", "missing", "configured scenario"),
        ("example", "cost", "missing", "configured scenario"),
        ("example", "path", 12, "configured scenario"),
    ],
)
def test_invalid_configuration_is_loud(
    document: dict[str, Any],
    section: str | None,
    key: str,
    value: object,
    message: str,
) -> None:
    target = document if section is None else document[section]
    target[key] = value
    with pytest.raises(ValueError, match=message):
        parse_config(document)


@pytest.mark.parametrize("value", [None, [], {1: "not a string key"}])
def test_non_object_configuration_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_config(value)


def test_missing_fields_and_bad_policy_cost_references(
    document: dict[str, Any],
) -> None:
    missing = copy.deepcopy(document)
    del missing["market"]["drift"]
    with pytest.raises(ValueError, match="missing fields"):
        parse_config(missing)
    for field in ("policies", "cost_schedules"):
        duplicate = copy.deepcopy(document)
        duplicate[field].append(duplicate[field][0])
        with pytest.raises(ValueError, match="unique"):
            parse_config(duplicate)
    document["policies"][0]["kind"] = "clairvoyant"
    with pytest.raises(ValueError, match=r"policy\.kind"):
        parse_config(document)


def test_cost_fields_must_be_explicit(document: dict[str, Any]) -> None:
    del document["cost_schedules"][0]["fee_bps"]
    with pytest.raises(ValueError, match="missing fields"):
        parse_config(document)


def test_optional_premium_and_defaults(document: dict[str, Any]) -> None:
    contract = document["contract"]
    contract["premium"] = 7.25
    for key in (
        "rate",
        "dividend_yield",
        "quantity",
        "multiplier",
        "initial_cash",
        "initial_inventory",
    ):
        del contract[key]
    parsed = parse_config(document)
    assert parsed.contract.premium == 7.25
    assert parsed.contract.rate == 0
    assert parsed.contract.quantity == -1
    assert parsed.contract.multiplier == 1


def test_all_artifacts_deterministic_and_path_independent(
    measured: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    original, payload = measured
    rerun = run_experiment(parse_config(payload["config"]), tmp_path / "other-location")
    assert payload == rerun
    assert set(payload["artifacts"]) == {path.name for path in original.iterdir()}
    for filename in payload["artifacts"]:
        assert (original / filename).read_bytes() == (
            tmp_path / "other-location" / filename
        ).read_bytes()
    on_disk = json.loads((original / "results.json").read_text(encoding="utf-8"))
    assert on_disk["summary"] == payload["summary"]
    assert str(original) not in json.dumps(on_disk)
    assert "timestamp" not in json.dumps(on_disk).lower()


def test_provenance_hashes_actual_sources_and_data(
    measured: tuple[Path, dict[str, Any]],
) -> None:
    _, payload = measured
    provenance = payload["provenance"]
    config = parse_config(payload["config"])
    assert provenance["config_sha256"] == _digest(config.to_dict())
    assert provenance["source"] == _source_fingerprint()
    assert provenance["source_sha256"] == _digest(provenance["source"]["module_sha256"])
    root = EXAMPLE.parent.parent / "src" / "quant_pricing"
    modules = provenance["source"]["module_sha256"]
    assert {
        "hedging.py",
        "hedge_paths.py",
        "hedge_experiment.py",
        "black_scholes.py",
    } <= modules.keys()
    for filename, digest in modules.items():
        assert not Path(filename).is_absolute()
        assert digest == hashlib.sha256((root / filename).read_bytes()).hexdigest()
    assert provenance["versions"]["numpy"] == np.__version__
    assert provenance["data_generation"]["rng"] == "PCG64"
    assert provenance["initial_forward"]["value"] == pytest.approx(
        100 * math.exp(0.03 * 0.25)
    )
    for seed in config.market.seeds:
        recovered_shocks = []
        for volatility in config.market.path_volatilities:
            spots = generate_gbm(
                GBMConfig(
                    config.market.spot,
                    config.market.drift,
                    volatility,
                    config.contract.maturity,
                    config.market.steps,
                    config.market.paths,
                    seed,
                )
            )
            key = f"seed={seed};vol={volatility!r}"
            assert (
                provenance["path_data_sha256"][key]
                == hashlib.sha256(
                    spots.astype("<f8", copy=False).tobytes(order="C")
                ).hexdigest()
            )
            dt = config.contract.maturity / config.market.steps
            recovered_shocks.append(
                (
                    np.diff(np.log(spots), axis=0)
                    - (config.market.drift - 0.5 * volatility**2) * dt
                )
                / (volatility * math.sqrt(dt))
            )
        for shocks in recovered_shocks[1:]:
            np.testing.assert_allclose(shocks, recovered_shocks[0], atol=2e-13)


def test_csv_summary_matches_retained_path_distribution(
    measured: tuple[Path, dict[str, Any]],
) -> None:
    directory, payload = measured
    rows = _csv(directory / "path_results.csv")
    summaries = _csv(directory / "summary.csv")
    assert len(rows) == 2 * 3 * 6 * 3 * 12
    assert len(summaries) == 3 * 6 * 3
    assert len(_csv(directory / "per_seed_summary.csv")) == 2 * 3 * 6 * 3
    for summary in summaries:
        scenario = [
            row
            for row in rows
            if all(
                row[key] == summary[key]
                for key in ("path_volatility", "policy", "cost")
            )
        ]
        nominal = np.array([float(row["final_pnl"]) for row in scenario])
        error = np.array([float(row["hedge_error"]) for row in scenario])
        assert int(summary["paths"]) == len(scenario) == 24
        assert float(summary["nominal_pnl_mean"]) == pytest.approx(nominal.mean())
        assert float(summary["nominal_pnl_sample_sd"]) == pytest.approx(
            nominal.std(ddof=1)
        )
        assert float(summary["discounted_error_bias"]) == pytest.approx(error.mean())
        assert float(summary["discounted_error_rmse"]) == pytest.approx(
            np.sqrt(np.mean(error**2))
        )
        assert float(summary["discounted_error_sample_sd"]) == pytest.approx(
            error.std(ddof=1)
        )
        for name, quantile in (("p05", 0.05), ("median", 0.5), ("p95", 0.95)):
            assert float(summary[f"nominal_pnl_{name}"]) == pytest.approx(
                np.quantile(nominal, quantile)
            )
        assert np.allclose(error, nominal * math.exp(-0.03 * 0.25))
    assert len(payload["summary"]) == len(summaries)


def test_cost_drag_is_paired_to_same_policy_and_includes_financing(
    measured: tuple[Path, dict[str, Any]],
) -> None:
    directory, _ = measured
    rows = _csv(directory / "path_results.csv")
    zero = {
        (row["seed"], row["path_volatility"], row["policy"], row["path"]): row
        for row in rows
        if row["cost"] == "zero"
    }
    observed_funding_drag = False
    for row in rows:
        reference = zero[
            (row["seed"], row["path_volatility"], row["policy"], row["path"])
        ]
        drag = float(row["paired_cost_drag"])
        assert drag == pytest.approx(
            float(reference["final_pnl"]) - float(row["final_pnl"])
        )
        explicit = sum(
            float(row[key]) for key in ("fees", "spread_cost", "slippage_cost")
        )
        funding = float(reference["financing"]) - float(row["financing"])
        assert drag == pytest.approx(explicit + funding, abs=1e-11)
        # Trade units are unchanged; side-price notional changes with execution costs.
        assert float(row["mean_abs_residual_delta"]) == float(
            reference["mean_abs_residual_delta"]
        )
        assert float(row["trade_count"]) == float(reference["trade_count"])
        components = (
            sum(
                float(row[key])
                for key in (
                    "option_pnl",
                    "hedge_pnl",
                    "financing",
                    "dividends",
                )
            )
            - explicit
        )
        assert float(row["final_pnl"]) == pytest.approx(components, abs=1e-10)
        observed_funding_drag |= funding > 1e-8
        if row["cost"] == "zero":
            assert drag == 0
    assert observed_funding_drag


def test_paired_uncertainty_uses_identical_seed_path_keys(
    measured: tuple[Path, dict[str, Any]],
) -> None:
    directory, payload = measured
    rows = _csv(directory / "path_results.csv")
    comparisons = payload["paired_comparisons"]
    assert len(comparisons) == 3 * 6 * 3 * 2 * 3
    indexed = {
        (
            float(row["path_volatility"]),
            row["policy"],
            row["cost"],
            int(row["seed"]),
            int(row["path"]),
        ): row
        for row in rows
    }
    config = parse_config(payload["config"])
    for comparison in comparisons:
        differences = []
        for seed in config.market.seeds:
            for path in range(config.market.paths):
                row = indexed[
                    (
                        comparison["path_volatility"],
                        comparison["policy"],
                        comparison["cost"],
                        seed,
                        path,
                    )
                ]
                reference = indexed[
                    (
                        comparison["reference_path_volatility"],
                        comparison["reference_policy"],
                        comparison["cost"],
                        seed,
                        path,
                    )
                ]
                metric = comparison["metric"]
                if metric == "squared_discounted_error":
                    delta = (
                        float(row["hedge_error"]) ** 2
                        - float(reference["hedge_error"]) ** 2
                    )
                else:
                    delta = float(row[metric]) - float(reference[metric])
                differences.append(delta)
        values = np.array(differences)
        assert comparison["paired_difference_mean"] == pytest.approx(values.mean())
        assert comparison["paired_difference_mean_se"] == pytest.approx(
            values.std(ddof=1) / math.sqrt(values.size)
        )
        assert comparison["reference_policy"] == (
            "periodic_1" if comparison["axis"] == "policy" else comparison["policy"]
        )


def test_selected_ledger_and_chart_are_measured(
    measured: tuple[Path, dict[str, Any]],
) -> None:
    directory, payload = measured
    ledger = _csv(directory / "example_ledger.csv")
    assert len(ledger) == 9
    trace = payload["config"]["example"]
    for row in ledger:
        for key in ("seed", "path_volatility", "policy", "cost", "path"):
            assert row[key] == str(trace[key])
    assert int(ledger[0]["step"]) == 0
    assert int(ledger[-1]["step"]) == 8
    assert float(ledger[-1]["hedge_inventory"]) == 0
    assert float(ledger[-1]["option_units"]) == 0
    assert float(ledger[-1]["unrealized_pnl"]) == pytest.approx(0)
    terminal = next(
        row
        for row in _csv(directory / "path_results.csv")
        if all(row[key] == str(trace[key]) for key in trace)
    )
    assert float(ledger[-1]["equity"]) == pytest.approx(float(terminal["final_pnl"]))
    root = ET.parse(directory / "hedging_tradeoff.svg").getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    circles = root.findall(".//svg:circle", namespace)
    assert len(circles) == len(payload["summary"])
    for point in circles:
        title = point.find("svg:title", namespace)
        assert title is not None and "paired drag=" in (title.text or "")
        assert math.isfinite(float(point.attrib["cx"]))
        assert math.isfinite(float(point.attrib["cy"]))


def test_nonfinite_result_statistics_fail_instead_of_filling() -> None:
    contract = HedgeContract("call", 100, 0.25, 0.2)
    spots = np.array([[100.0, 100.0], [102.0, 98.0]])
    result = simulate_hedge(
        contract, prepare_market(contract, spots), HedgePolicy("periodic"), HedgeCosts()
    )
    arrays = _arrays(result, result)
    bad = replace(result, final_pnl=np.array([1e308, -1e308]))
    zero = replace(result, final_pnl=np.array([-1e308, 1e308]))
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="nonfinite result"),
    ):
        _arrays(bad, zero)
    bad_shape = replace(result, final_pnl=np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError, match="result array"):
        _arrays(bad_shape, result)
    arrays["hedge_error"] = np.array([1e300, -1e300])
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="nonfinite summary"),
    ):
        _summarize(arrays)


def test_zero_turnover_zero_dispersion_and_no_periodic_reference(
    document: dict[str, Any],
    tmp_path: Path,
) -> None:
    document["contract"]["quantity"] = 0.0
    document["market"]["path_volatilities"] = [0.12]
    document["policies"] = [{"id": "none", "kind": "none"}]
    document["cost_schedules"].reverse()
    document["example"].update(policy="none", path_volatility=0.12, cost="zero")
    result = run_experiment(parse_config(document), tmp_path)
    for row in result["summary"]:
        assert row["discounted_error_sample_sd"] == 0
        assert row["turnover_mean"] == 0
        assert row["paired_cost_drag_mean"] == 0
    for row in result["paired_comparisons"]:
        assert row["reference_policy"] == "none"
        assert row["paired_difference_mean"] == 0
    assert "nan" not in (tmp_path / "hedging_tradeoff.svg").read_text().lower()


def test_config_identity_tracks_economic_changes(document: dict[str, Any]) -> None:
    first = parse_config(document)
    document["market"]["drift"] = 0.04
    second = parse_config(document)
    assert _digest(first.to_dict()) != _digest(second.to_dict())
    assert _digest({"a": 1, "b": 2}) == _digest({"b": 2, "a": 1})


def test_cli_real_module_entrypoint(document: dict[str, Any], tmp_path: Path) -> None:
    document["market"].update(paths=2, steps=1, seeds=[1729], path_volatilities=[0.2])
    document["policies"] = [document["policies"][1]]
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    destination = tmp_path / "outputs"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_pricing.hedge_experiment",
            str(config_path),
            "--output-dir",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Wrote 7 artifacts" in completed.stdout
    assert (destination / "results.json").is_file()
    assert main([str(config_path), "--output-dir", str(destination)]) == 0


def test_cli_invalid_config_reports_error_without_results(
    document: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document["zero_cost_reference"] = "baseline"
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")
    destination = tmp_path / "outputs"
    with pytest.raises(SystemExit) as error:
        main([str(config_path), "--output-dir", str(destination)])
    assert error.value.code == 2
    assert "exactly zero" in capsys.readouterr().err
    assert not destination.exists()
    with pytest.raises(SystemExit):
        main([str(tmp_path / "absent.json"), "--output-dir", str(destination)])


@pytest.mark.parametrize("rate", [1e4, -1e4])
def test_forward_overflow_is_loud_before_artifact_creation(
    document: dict[str, Any],
    tmp_path: Path,
    rate: float,
) -> None:
    document["contract"]["rate"] = rate
    config = parse_config(document)
    with pytest.raises((ValueError, OverflowError)):
        run_experiment(config, tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_comparison_reference_nearest_vol_is_explicit(document: dict[str, Any]) -> None:
    document["market"]["path_volatilities"] = [0.12, 0.32]
    document["example"]["path_volatility"] = 0.12
    config = parse_config(document)
    arrays = {"final_pnl": np.array([1.0, 2.0]), "hedge_error": np.array([1.0, 2.0])}
    pooled = {
        (vol, policy.id, cost.id): arrays
        for vol in config.market.path_volatilities
        for policy in config.policies
        for cost in config.cost_schedules
    }
    rows = _comparisons(config, pooled)
    assert all(
        row["reference_path_volatility"] == 0.12
        for row in rows
        if row["axis"] == "path_volatility"
    )
