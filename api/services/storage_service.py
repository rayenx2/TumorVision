import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from api.schemas.feedback import FeedbackRequest, HistoryItem
from api.schemas.prediction import PredictionMetadata
from src.utils.logger import api_logger as logger


class StorageService:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    uncertainty_score REAL NOT NULL,
                    is_uncertain INTEGER NOT NULL,
                    inference_time_ms REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedbacks (
                    prediction_id TEXT PRIMARY KEY,
                    correct_class TEXT NOT NULL,
                    comment TEXT,
                    radiologist_name TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                );
                """
            )

        logger.info("Database initialized at %s", self.db_path)

    def save_prediction(self, metadata: PredictionMetadata) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO predictions (
                    prediction_id,
                    predicted_class,
                    confidence,
                    uncertainty_score,
                    is_uncertain,
                    inference_time_ms,
                    model_version,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    metadata.prediction_id,
                    metadata.predicted_class,
                    metadata.confidence,
                    metadata.uncertainty_score,
                    int(metadata.is_uncertain),
                    metadata.inference_time_ms,
                    metadata.model_version,
                    metadata.timestamp.isoformat(),
                ),
            )

    def get_prediction(self, prediction_id: str) -> PredictionMetadata | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM predictions
                WHERE prediction_id = ?;
                """,
                (prediction_id,),
            ).fetchone()

        if row is None:
            return None

        return PredictionMetadata(
            prediction_id=row["prediction_id"],
            predicted_class=row["predicted_class"],
            confidence=row["confidence"],
            uncertainty_score=row["uncertainty_score"],
            is_uncertain=bool(row["is_uncertain"]),
            inference_time_ms=row["inference_time_ms"],
            model_version=row["model_version"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
        )

    def list_predictions(self, limit: int, offset: int) -> tuple[list[HistoryItem], int]:
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
            rows = connection.execute(
                """
                SELECT
                    predictions.prediction_id,
                    predictions.predicted_class,
                    predictions.confidence,
                    predictions.uncertainty_score,
                    predictions.is_uncertain,
                    predictions.inference_time_ms,
                    predictions.model_version,
                    predictions.timestamp,
                    feedbacks.correct_class
                FROM predictions
                LEFT JOIN feedbacks
                    ON predictions.prediction_id = feedbacks.prediction_id
                ORDER BY predictions.timestamp DESC
                LIMIT ? OFFSET ?;
                """,
                (limit, offset),
            ).fetchall()

        items = [
            HistoryItem(
                prediction_id=row["prediction_id"],
                predicted_class=row["predicted_class"],
                confidence=row["confidence"],
                uncertainty_score=row["uncertainty_score"],
                is_uncertain=bool(row["is_uncertain"]),
                inference_time_ms=row["inference_time_ms"],
                model_version=row["model_version"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                feedback_received=row["correct_class"] is not None,
                correct_class=row["correct_class"],
            )
            for row in rows
        ]

        return items, total

    def save_feedback(self, feedback: FeedbackRequest) -> bool:
        with self._connect() as connection:
            prediction_exists = connection.execute(
                """
                SELECT 1
                FROM predictions
                WHERE prediction_id = ?;
                """,
                (feedback.prediction_id,),
            ).fetchone()

            if prediction_exists is None:
                return False

            connection.execute(
                """
                INSERT OR REPLACE INTO feedbacks (
                    prediction_id,
                    correct_class,
                    comment,
                    radiologist_name,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    feedback.prediction_id,
                    feedback.correct_class,
                    feedback.comment,
                    feedback.radiologist_name,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        return True
