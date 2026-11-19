import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"


def main():
    if not METRICS_PATH.exists():
        raise FileNotFoundError("Run `python -m src.train_model` first.")

    with open(METRICS_PATH, encoding="utf-8") as f:
        report = json.load(f)

    print("\nCUSTOMER CHURN MODEL EVALUATION")
    print("=" * 72)
    print(f"Selected model: {report['selected_model']}")
    print(f"Test records: {report['test_records']}")
    print(f"Prediction threshold: {report['prediction_threshold']}")
    print("-" * 72)

    for name, m in report["results"].items():
        print(f"\n{name}")
        print(f"  CV ROC-AUC : {m['cv_roc_auc']:.4f}")
        print(f"  Test ROC-AUC: {m['test_roc_auc']:.4f}")
        print(f"  PR-AUC      : {m['pr_auc']:.4f}")
        print(f"  Precision   : {m['precision']:.4f}")
        print(f"  Recall      : {m['recall']:.4f}")
        print(f"  F1-score    : {m['f1']:.4f}")
        print(f"  Accuracy    : {m['accuracy']:.4f}")
        print(f"  Confusion matrix: {m['confusion_matrix']}")

    print("\nBusiness note: review recall, precision, PR-AUC and the cost of")
    print("retention offers versus missed churners before choosing a production threshold.")


if __name__ == "__main__":
    main()
