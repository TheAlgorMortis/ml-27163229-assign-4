"""Resumable experiment helpers for the Assignment 4 notebook."""

import hashlib
import json
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)


def log(message):
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _fingerprint_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(root, data_path, feature_set, feature_names, dev_idx, test_idx):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "data_sha256": _fingerprint_file(data_path),
        "feature_set": feature_set,
        "feature_names": list(feature_names),
        "dev_indices_sha256": hashlib.sha256(np.asarray(dev_idx, dtype=np.int64).tobytes()).hexdigest(),
        "test_indices_sha256": hashlib.sha256(np.asarray(test_idx, dtype=np.int64).tobytes()).hexdigest(),
        "preprocessing": "Assignment 2 tree: service=MISSING; state no=NOT_USED; numeric passthrough; one-hot proto/state/service",
        "sklearn_version": sklearn.__version__,
    }
    path = root / "manifest.json"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous != manifest:
            raise ValueError(f"Saved run at {root} uses different data, split, features, preprocessing, or sklearn version. Choose a new run directory.")
    else:
        _atomic_json(path, manifest)
    for folder in ("rf", "xgb", "final", "models"):
        (root / folder).mkdir(exist_ok=True)
    log(f"Run directory ready: {root.resolve()}")
    return root


def candidate_id(family, config):
    encoded = json.dumps({"family": family, "config": config}, sort_keys=True).encode()
    return f"{family}-{hashlib.sha256(encoded).hexdigest()[:12]}"


def completed_results(folder):
    rows = []
    for path in sorted(Path(folder).glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            row = json.load(stream)
        if row.get("status") == "completed":
            rows.append(row)
    return rows


def _candidate_paths(root, family, identifier):
    folder = Path(root) / family
    return folder / f"{identifier}.json", folder / f"{identifier}.npz"


def rf_grid(class_weights=(None, "balanced_subsample"), max_features=("sqrt", 0.25, 0.5), min_samples_leaf=(1, 5, 10, 20)):
    return [
        {"class_weight": weight, "max_features": features, "min_samples_leaf": leaf}
        for weight, features, leaf in product(class_weights, max_features, min_samples_leaf)
    ]


def run_rf_candidates(root, X, y, configurations, n_estimators, n_jobs=4):
    y_array = np.asarray(y)
    classes = np.unique(y_array)
    configurations = list(configurations)
    for number, partial in enumerate(configurations, start=1):
        config = {**partial, "n_estimators": int(n_estimators)}
        identifier = candidate_id("rf", config)
        result_path, prediction_path = _candidate_paths(root, "rf", identifier)
        if result_path.exists() and prediction_path.exists():
            log(f"RF {number}/{len(configurations)} skip completed {identifier}: {config}")
            continue
        log(f"RF {number}/{len(configurations)} start {identifier}: {config}")
        started = time.perf_counter()
        model = RandomForestClassifier(
            **config, bootstrap=True, oob_score=True, random_state=42,
            n_jobs=n_jobs,
        )
        model.fit(X, y_array)
        probabilities = np.asarray(model.oob_decision_function_, dtype=np.float32)
        valid = np.isfinite(probabilities).all(axis=1) & (probabilities.sum(axis=1) > 0)
        if not valid.all():
            log(f"RF {identifier}: {np.count_nonzero(~valid)} rows lacked usable OOB votes")
        if not valid.any():
            raise RuntimeError(f"RF {identifier} produced no usable OOB predictions")
        predictions = classes[np.argmax(probabilities[valid], axis=1)]
        result = {
            "status": "completed", "family": "rf", "id": identifier,
            "config": config, "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "oob_macro_f1": float(f1_score(y_array[valid], predictions, average="macro", labels=classes, zero_division=0)),
            "oob_accuracy": float(accuracy_score(y_array[valid], predictions)),
            "oob_coverage": float(valid.mean()),
            "probability_path": str(prediction_path),
        }
        _atomic_npz(prediction_path, probabilities=probabilities, valid=valid)
        _atomic_json(result_path, result)
        log(f"RF {number}/{len(configurations)} done {identifier}: OOB macro-F1={result['oob_macro_f1']:.4f}; {result['elapsed_seconds']/60:.1f} min")
    return completed_results(Path(root) / "rf")


def rf_finalist_configs(results, screening_trees):
    chosen = []
    for weight in (None, "balanced_subsample"):
        arm = [
            row for row in results
            if row["config"]["n_estimators"] == screening_trees
            and row["config"]["class_weight"] == weight
        ]
        if not arm:
            raise RuntimeError(f"No completed RF screening results for class_weight={weight!r}")
        winner = max(arm, key=lambda row: (row["oob_macro_f1"], row["id"]))
        chosen.append({key: winner["config"][key] for key in ("class_weight", "max_features", "min_samples_leaf")})
    return chosen


def selected_rf_results(results, final_trees):
    selected = []
    for weight in (None, "balanced_subsample"):
        arm = [
            row for row in results
            if row["config"]["n_estimators"] == final_trees
            and row["config"]["class_weight"] == weight
        ]
        if not arm:
            raise RuntimeError(f"No completed {final_trees}-tree RF result for class_weight={weight!r}")
        selected.append(max(arm, key=lambda row: (row["oob_macro_f1"], row["id"])))
    return selected


def xgb_grid(max_depth=(3, 6), min_child_weight=(1, 10), learning_rate=(0.05, 0.1)):
    return [
        {"max_depth": depth, "min_child_weight": child, "learning_rate": rate}
        for depth, child, rate in product(max_depth, min_child_weight, learning_rate)
    ]


def _xgb_classifier(config, n_estimators, n_jobs, early_stopping_rounds=None, callbacks=None):
    return xgboost.XGBClassifier(
        **config, n_estimators=n_estimators, objective="multi:softprob",
        tree_method="hist", eval_metric="mlogloss", subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
        n_jobs=n_jobs, early_stopping_rounds=early_stopping_rounds,
        callbacks=callbacks,
    )


def run_xgb_candidates(root, X, y, fit_idx, validation_idx, configurations, max_rounds=600, patience=30, n_jobs=4):
    class TimestampProgress(xgboost.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if (epoch + 1) % 50 == 0 or epoch == 0:
                losses = evals_log.get("validation_0", {}).get("mlogloss", [])
                loss_text = f"; validation mlogloss={losses[-1]:.5f}" if losses else ""
                log(f"XGB boosting round {epoch + 1}/{max_rounds}{loss_text}")
            return False

    y_array = np.asarray(y)
    classes = np.unique(y_array)
    configurations = list(configurations)
    for number, config in enumerate(configurations, start=1):
        full_config = {
            **config, "max_rounds": int(max_rounds), "patience": int(patience),
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        }
        identifier = candidate_id("xgb", full_config)
        result_path, prediction_path = _candidate_paths(root, "xgb", identifier)
        if result_path.exists() and prediction_path.exists():
            log(f"XGB {number}/{len(configurations)} skip completed {identifier}: {config}")
            continue
        log(f"XGB {number}/{len(configurations)} start {identifier}: {config}")
        started = time.perf_counter()
        model = _xgb_classifier(
            config, max_rounds, n_jobs, early_stopping_rounds=patience,
            callbacks=[TimestampProgress()],
        )
        model.fit(
            X[fit_idx], y_array[fit_idx],
            eval_set=[(X[validation_idx], y_array[validation_idx])],
            verbose=False,
        )
        probabilities = np.asarray(model.predict_proba(X[validation_idx]), dtype=np.float32)
        predictions = classes[np.argmax(probabilities, axis=1)]
        best_iteration = int(model.best_iteration)
        result = {
            "status": "completed", "family": "xgb", "id": identifier,
            "config": full_config, "xgboost_version": xgboost.__version__,
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": float(time.perf_counter() - started),
            "best_iteration": best_iteration,
            "validation_macro_f1": float(f1_score(y_array[validation_idx], predictions, average="macro", labels=classes, zero_division=0)),
            "validation_accuracy": float(accuracy_score(y_array[validation_idx], predictions)),
            "validation_mlogloss": float(model.best_score),
            "probability_path": str(prediction_path),
        }
        _atomic_npz(prediction_path, probabilities=probabilities)
        _atomic_json(result_path, result)
        log(f"XGB {number}/{len(configurations)} done {identifier}: validation macro-F1={result['validation_macro_f1']:.4f}; best round={best_iteration + 1}; {result['elapsed_seconds']/60:.1f} min")
    return completed_results(Path(root) / "xgb")


def selected_xgb_result(results):
    if not results:
        raise RuntimeError("No completed XGBoost validation results")
    return max(results, key=lambda row: (row["validation_macro_f1"], row["id"]))


def _metrics(y_true, y_pred, probabilities, classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    attack_true = y_true != 0
    attack_pred = y_pred != 0
    tp = int(np.sum(attack_true & attack_pred))
    fp = int(np.sum(~attack_true & attack_pred))
    tn = int(np.sum(~attack_true & ~attack_pred))
    fn = int(np.sum(attack_true & ~attack_pred))
    normal_column = int(np.flatnonzero(classes == 0)[0])
    attack_score = 1.0 - probabilities[:, normal_column]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "attack_recall": float(recall_score(attack_true, attack_pred, zero_division=0)),
        "attack_precision": float(precision_score(attack_true, attack_pred, zero_division=0)),
        "attack_f1": float(f1_score(attack_true, attack_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)),
        "true_positives": tp, "false_positives": fp,
        "true_negatives": tn, "false_negatives": fn,
        "attack_roc_auc": float(roc_auc_score(attack_true, attack_score)),
        "attack_average_precision": float(average_precision_score(attack_true, attack_score)),
    }


def run_final_models(root, X_rf, X_xgb, X_test_rf, X_test_xgb, y_train, y_test, rf_results, xgb_result, n_jobs=4):
    """Lock finalist IDs before first held-out evaluation; resume safely."""
    root = Path(root)
    (root / "final").mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    selected = {
        "random_forest": rf_results[0]["id"],
        "weighted_random_forest": rf_results[1]["id"],
        "xgboost": xgb_result["id"],
    }
    lock_path = root / "final" / "selection.lock.json"
    if lock_path.exists():
        with lock_path.open(encoding="utf-8") as stream:
            locked = json.load(stream)
        if locked != selected:
            raise RuntimeError("Final-test selection is already locked to different candidates. Do not retune using test feedback; inspect selection.lock.json before proceeding.")
    else:
        _atomic_json(lock_path, selected)
        log("Locked finalists before evaluating the held-out test set")

    classes = np.unique(np.asarray(y_train))
    outputs = []
    for name, result in zip(("random_forest", "weighted_random_forest", "xgboost"), (*rf_results, xgb_result)):
        result_path = root / "final" / f"{name}.json"
        prediction_path = root / "final" / f"{name}.npz"
        if result_path.exists() and prediction_path.exists():
            with result_path.open(encoding="utf-8") as stream:
                outputs.append(json.load(stream))
            log(f"Final {name}: loaded saved test result")
            continue
        log(f"Final {name}: fitting selected configuration")
        started = time.perf_counter()
        if name == "xgboost":
            config = {key: result["config"][key] for key in ("max_depth", "min_child_weight", "learning_rate")}
            model = _xgb_classifier(config, result["best_iteration"] + 1, n_jobs)
            model.fit(X_xgb, np.asarray(y_train), verbose=False)
            model_path = root / "models" / "xgboost.ubj"
            temporary = model_path.with_name("xgboost.tmp.ubj")
            model.save_model(str(temporary))
            os.replace(temporary, model_path)
            probabilities = np.asarray(model.predict_proba(X_test_xgb), dtype=np.float32)
        else:
            model = RandomForestClassifier(
                **result["config"], bootstrap=True, oob_score=True,
                random_state=42, n_jobs=n_jobs,
            )
            model.fit(X_rf, np.asarray(y_train))
            model_path = root / "models" / f"{name}.joblib"
            temporary = model_path.with_name(model_path.name + ".tmp")
            joblib.dump(model, temporary)
            os.replace(temporary, model_path)
            probabilities = np.asarray(model.predict_proba(X_test_rf), dtype=np.float32)
        predictions = classes[np.argmax(probabilities, axis=1)]
        metrics = _metrics(y_test, predictions, probabilities, classes)
        record = {
            "model": name, "selected_id": result["id"],
            "model_path": str(model_path),
            "prediction_path": str(prediction_path),
            "elapsed_seconds": float(time.perf_counter() - started),
            "completed_at": datetime.now().astimezone().isoformat(),
            **metrics,
        }
        _atomic_npz(prediction_path, predictions=predictions, probabilities=probabilities)
        _atomic_json(result_path, record)
        outputs.append(record)
        log(f"Final {name}: macro-F1={record['macro_f1']:.4f}, attack recall={record['attack_recall']:.4f}, FPR={record['false_positive_rate']:.4f}; {record['elapsed_seconds']/60:.1f} min")
    return pd.DataFrame(outputs)


def load_final_table(root):
    rows = []
    for name in ("random_forest", "weighted_random_forest", "xgboost"):
        path = Path(root) / "final" / f"{name}.json"
        if path.exists():
            with path.open(encoding="utf-8") as stream:
                rows.append(json.load(stream))
    return pd.DataFrame(rows)


def saved_prediction_diagnostics(root, y_test, figure_path=None):
    """Describe completed held-out predictions without fitting or selecting models."""
    import matplotlib.pyplot as plt

    root = Path(root)
    folder = root / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_test)
    classes = np.unique(y_true)
    names = ("random_forest", "weighted_random_forest", "xgboost")
    class_rows = []
    error_rows = []
    runtime_rows = []
    matrices = {}

    for name in names:
        result_path = root / "final" / f"{name}.json"
        prediction_path = root / "final" / f"{name}.npz"
        if not (result_path.exists() and prediction_path.exists()):
            raise FileNotFoundError(f"Complete the final evaluation before diagnosing {name}")
        with result_path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        with np.load(prediction_path) as arrays:
            y_pred = arrays["predictions"]
        if len(y_pred) != len(y_true):
            raise ValueError(f"Saved predictions for {name} do not match the test labels")
        recalculated = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        if not np.isclose(recalculated, saved["macro_f1"], rtol=0, atol=1e-10):
            raise ValueError(f"Saved predictions for {name} do not match the recorded score; check test-row order")

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=classes, zero_division=0,
        )
        counts = confusion_matrix(y_true, y_pred, labels=classes)
        normal_index = int(np.flatnonzero(classes == 0)[0])
        false_positives = int(counts[normal_index].sum() - counts[normal_index, normal_index])
        false_negatives = int(counts[:, normal_index].sum() - counts[normal_index, normal_index])
        if (false_positives != saved["false_positives"] or
                false_negatives != saved["false_negatives"]):
            raise ValueError(f"Saved predictions for {name} do not match the recorded binary errors")
        matrices[name] = counts
        pd.DataFrame(counts, index=classes, columns=classes).to_csv(
            folder / f"{name}_confusion_counts.csv", index_label="true_class"
        )
        normalized = np.divide(
            counts, counts.sum(axis=1, keepdims=True),
            out=np.zeros_like(counts, dtype=float), where=counts.sum(axis=1, keepdims=True) != 0,
        )
        pd.DataFrame(normalized, index=classes, columns=classes).to_csv(
            folder / f"{name}_confusion_row_normalized.csv", index_label="true_class"
        )
        for index, label in enumerate(classes):
            class_rows.append({
                "model": name, "class": int(label), "support": int(support[index]),
                "precision": float(precision[index]), "recall": float(recall[index]),
                "f1": float(f1[index]),
            })
            row = {
                "model": name, "class": int(label), "support": int(support[index]),
                "correct_subtype": int(counts[index, index]),
            }
            if label == 0:
                row["predicted_normal"] = int(counts[index, index])
                row["predicted_attack"] = int(counts[index].sum() - counts[index, index])
                row["missed_as_normal"] = 0
                row["wrong_attack_subtype"] = 0
            else:
                row["predicted_normal"] = int(counts[index, normal_index])
                row["predicted_attack"] = int(counts[index].sum() - counts[index, normal_index])
                row["missed_as_normal"] = int(counts[index, normal_index])
                row["wrong_attack_subtype"] = int(
                    counts[index].sum() - counts[index, normal_index] - counts[index, index]
                )
            error_rows.append(row)
        runtime_rows.append({
            "model": name, "final_fit_predict_save_minutes": saved["elapsed_seconds"] / 60,
            "macro_f1": saved["macro_f1"], "attack_recall": saved["attack_recall"],
            "false_positive_rate": saved["false_positive_rate"],
        })
        log(f"Diagnosed {name} from saved predictions; no fitting performed")

    per_class = pd.DataFrame(class_rows)
    errors = pd.DataFrame(error_rows)
    runtimes = pd.DataFrame(runtime_rows)
    per_class.to_csv(folder / "per_class_metrics.csv", index=False)
    errors.to_csv(folder / "error_breakdown.csv", index=False)
    runtimes.to_csv(folder / "final_runtime.csv", index=False)

    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.5), constrained_layout=True)
        titles = ("Unweighted forest", "Weighted forest", "Boosting")
        for axis, name, title in zip(axes, names, titles):
            counts = matrices[name]
            normalized = counts / counts.sum(axis=1, keepdims=True)
            image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            axis.set_title(title)
            axis.set_xticks(np.arange(len(classes)), classes)
            axis.set_yticks(np.arange(len(classes)), classes)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            for row in range(len(classes)):
                for column in range(len(classes)):
                    axis.text(
                        column, row, f"{counts[row, column]:d}",
                        ha="center", va="center", fontsize=8,
                        color="white" if normalized[row, column] >= 0.55 else "#202020",
                    )
        fig.colorbar(image, ax=axes, label="Fraction of true class", shrink=0.8)
        fig.savefig(figure_path, bbox_inches="tight")
        fig.savefig(figure_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved row-normalized confusion matrices: {figure_path}")
    return per_class, errors, runtimes
