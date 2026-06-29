from datetime import datetime, timezone

from supabase import Client, create_client

from api.schemas.feedback import FeedbackRequest, HistoryItem
from api.schemas.prediction import PredictionMetadata
from src.utils.logger import api_logger as logger


class SupabaseStorageService:
    def __init__(self, url: str, key: str) -> None:
        self.client: Client = create_client(url, key)
        logger.info("Supabase client initialized")

    def save_prediction(self, metadata: PredictionMetadata) -> None:
        try:
            data = {
                "prediction_id": metadata.prediction_id,
                "predicted_class": metadata.predicted_class,
                "confidence": metadata.confidence,
                "uncertainty_score": metadata.uncertainty_score,
                "is_uncertain": metadata.is_uncertain,
                "inference_time_ms": metadata.inference_time_ms,
                "model_version": metadata.model_version,
                "timestamp": metadata.timestamp.isoformat(),
            }
            self.client.table("predictions").upsert(data).execute()
        except Exception as e:
            logger.error("Failed to save prediction to Supabase: %s", e)
            raise

    def get_prediction(self, prediction_id: str) -> PredictionMetadata | None:
        try:
            response = (
                self.client.table("predictions")
                .select("*")
                .eq("prediction_id", prediction_id)
                .execute()
            )
            if not response.data:
                return None

            row = response.data[0]
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
        except Exception as e:
            logger.error("Failed to get prediction from Supabase: %s", e)
            raise

    def list_predictions(self, limit: int, offset: int) -> tuple[list[HistoryItem], int]:
        try:
            # Count total
            count_response = (
                self.client.table("predictions")
                .select("prediction_id", count="exact", head=True)
                .execute()
            )
            total = count_response.count if count_response.count is not None else 0

            # Query with join
            response = (
                self.client.table("predictions")
                .select("*, feedbacks(correct_class)")
                .order("timestamp", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            items = []
            for row in response.data:
                feedbacks = row.get("feedbacks")
                correct_class = None
                if isinstance(feedbacks, list) and len(feedbacks) > 0:
                    correct_class = feedbacks[0].get("correct_class")
                elif isinstance(feedbacks, dict):
                    correct_class = feedbacks.get("correct_class")

                items.append(
                    HistoryItem(
                        prediction_id=row["prediction_id"],
                        predicted_class=row["predicted_class"],
                        confidence=row["confidence"],
                        uncertainty_score=row["uncertainty_score"],
                        is_uncertain=bool(row["is_uncertain"]),
                        inference_time_ms=row["inference_time_ms"],
                        model_version=row["model_version"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        feedback_received=correct_class is not None,
                        correct_class=correct_class,
                    )
                )

            return items, total
        except Exception as e:
            logger.error("Failed to list predictions from Supabase: %s", e)
            raise

    def save_feedback(self, feedback: FeedbackRequest) -> bool:
        try:
            # Check if prediction exists
            check_response = (
                self.client.table("predictions")
                .select("prediction_id")
                .eq("prediction_id", feedback.prediction_id)
                .execute()
            )
            if not check_response.data:
                return False

            data = {
                "prediction_id": feedback.prediction_id,
                "correct_class": feedback.correct_class,
                "comment": feedback.comment,
                "radiologist_name": feedback.radiologist_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.client.table("feedbacks").upsert(data).execute()
            return True
        except Exception as e:
            logger.error("Failed to save feedback to Supabase: %s", e)
            raise
