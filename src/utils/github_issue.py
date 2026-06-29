import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests

from src.utils.logger import logger


class GitHubIssueCreator:
    """
    Automatically opens a GitHub Issue when model drift is detected.
    Uses the GitHub REST API with a token from environment variables.
    """

    GITHUB_API_URL = "https://api.github.com"

    def __init__(self):
        self.token = os.environ.get("GITHUB_TOKEN")
        self.repo = os.environ.get("GITHUB_REPOSITORY")  # e.g. "yourusername/kidney-tumor-system"

        if not self.token:
            raise EnvironmentError("GITHUB_TOKEN environment variable not set.")
        if not self.repo:
            raise EnvironmentError("GITHUB_REPOSITORY environment variable not set.")

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _build_issue_body(self, drift_summary: dict) -> str:
        """Build a detailed, readable issue body from the drift summary."""

        drift_score = drift_summary.get("drift_score", 0)
        threshold = drift_summary.get("drift_threshold", 0.15)
        drifted_features = drift_summary.get("drifted_features", [])
        reference_samples = drift_summary.get("reference_samples", "N/A")
        current_samples = drift_summary.get("current_samples", "N/A")
        timestamp = drift_summary.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

        # Build drifted features list
        if drifted_features:
            features_md = "\n".join([f"- `{f}`" for f in drifted_features])
        else:
            features_md = "- No specific features identified (dataset-level drift)"

        body = f"""## 🚨 Model Data Drift Detected

Automated drift monitoring has detected a significant shift in the distribution
of incoming CT scan image features compared to the training reference data.
**Human review and potential retraining is recommended.**

---

### 📊 Drift Summary

| Metric | Value |
|--------|-------|
| **Drift Score** | `{drift_score:.4f}` |
| **Threshold** | `{threshold}` |
| **Status** | 🔴 Drift Detected |
| **Reference Samples** | {reference_samples} |
| **Current Samples** | {current_samples} |
| **Detected At** | {timestamp} |

---

### 📉 Drifted Features

The following image features showed statistically significant distribution shift:

{features_md}

**Feature descriptions:**
- `mean_intensity` — average pixel brightness across CT scan
- `std_intensity` — variation in pixel values (texture)
- `contrast` — CLAHE contrast response
- `brightness_p25` / `brightness_p75` — brightness distribution quartiles

---

### ✅ Recommended Actions

- [ ] Review the drift HTML report in the `reports/drift/` folder of this run
- [ ] Check if new CT scans are from a different machine or hospital setting
- [ ] Collect additional labeled samples from the drifted distribution
- [ ] Run `make retrain` to trigger the retraining pipeline
- [ ] Monitor model accuracy on recent predictions
- [ ] Close this issue once retraining and validation is complete

---

### 🔁 How to Retrain

```bash
# Pull latest data
dvc pull

# Add new samples to data/raw/ and update DVC
dvc add data/raw/

# Run full pipeline
dvc repro

# Check evaluation results
cat reports/evaluation/metrics.json

# If AUC >= 0.90, push new model
python src/pipeline/stage_06_model_evaluation.py
```

---

*This issue was opened automatically by the weekly drift monitoring workflow.*
*Workflow: `.github/workflows/retrain.yml`*
"""
        return body

    def open_issue(self, drift_summary: dict) -> dict:
        """Open a GitHub Issue with drift details. Returns the created issue data."""

        drift_score = drift_summary.get("drift_score", 0)
        timestamp = drift_summary.get("timestamp", "")

        title = (
            f"🚨 Data Drift Detected — Score: {drift_score:.3f} "
            f"(threshold: {drift_summary.get('drift_threshold', 0.15)}) "
            f"[{timestamp[:8]}]"
        )

        body = self._build_issue_body(drift_summary)

        labels = ["drift-detected", "model-health", "automated"]
        assignees = []  # Add GitHub usernames here if needed e.g. ["yourusername"]

        payload = {
            "title": title,
            "body": body,
            "labels": labels,
            "assignees": assignees,
        }

        url = f"{self.GITHUB_API_URL}/repos/{self.repo}/issues"

        logger.info(f"Opening GitHub Issue in repo: {self.repo}")

        response = requests.post(url, headers=self.headers, json=payload, timeout=30)

        if response.status_code == 201:
            issue_data = response.json()
            logger.info(
                f"GitHub Issue opened successfully: #{issue_data['number']} "
                f"— {issue_data['html_url']}"
            )
            return issue_data
        else:
            logger.error(
                f"Failed to open GitHub Issue. "
                f"Status: {response.status_code} | Response: {response.text}"
            )
            response.raise_for_status()

    def check_duplicate_issue(self) -> bool:
        """
        Check if a drift issue is already open.
        Prevents duplicate issues from being created every week.
        """
        url = f"{self.GITHUB_API_URL}/repos/{self.repo}/issues"
        params = {
            "state": "open",
            "labels": "drift-detected",
            "per_page": 10,
        }

        response = requests.get(url, headers=self.headers, params=params, timeout=30)

        if response.status_code == 200:
            open_issues = response.json()
            if open_issues:
                logger.info(
                    f"Drift issue already open: #{open_issues[0]['number']} "
                    "— skipping duplicate creation."
                )
                return True
        return False


def run_drift_check_and_notify():
    """
    Entry point called by GitHub Actions.
    Reads drift_summary.json and opens an issue if drift is detected.
    """
    summary_path = Path("reports/drift/drift_summary.json")

    if not summary_path.exists():
        logger.error(f"Drift summary not found at {summary_path}")
        sys.exit(1)

    with open(summary_path) as f:
        drift_summary = json.load(f)

    drift_detected = drift_summary.get("drift_detected", False)

    if not drift_detected:
        logger.info("No drift detected. No GitHub Issue needed.")
        sys.exit(0)

    logger.info(
        f"Drift detected (score={drift_summary['drift_score']:.4f}). " "Opening GitHub Issue..."
    )

    creator = GitHubIssueCreator()

    # Avoid duplicate open issues
    if creator.check_duplicate_issue():
        logger.info("Skipping — drift issue already open.")
        sys.exit(0)

    issue = creator.open_issue(drift_summary)
    print(f"Issue URL: {issue['html_url']}")


if __name__ == "__main__":
    run_drift_check_and_notify()
