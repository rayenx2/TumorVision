from src.components.drift_detector import DriftDetector

detector = DriftDetector(
    reference_data_path="data/reference_features.csv",
    current_data_path="data/current_features.csv",
    drift_threshold=0.6,
    report_output_dir="reports/drift",
)

report = detector.run()

print(f"Drift detected: {report.drift_detected}")
print(f"Drift score: {report.drift_score:.4f}")
print(f"Drifted features: {report.drifted_features}")
