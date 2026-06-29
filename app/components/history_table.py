import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")


def render_history_table():
    st.header("Case History")

    try:
        response = requests.get(f"{API_URL}/records/history", params={"limit": 50, "offset": 0})
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])

            if not items:
                st.info("No prediction history available.")
                return

            # Format data for dataframe
            table_data = []
            for item in items:
                # Parse timestamp safely
                try:
                    ts = datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00"))
                    formatted_time = ts.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    formatted_time = item["timestamp"]

                table_data.append(
                    {
                        "Date": formatted_time,
                        "ID": item["prediction_id"][:8] + "...",
                        "Predicted": item["predicted_class"],
                        "Confidence": f"{item['confidence']*100:.1f}%",
                        "Uncertainty": (
                            f"{item['uncertainty_score']:.2f}" if item["is_uncertain"] else "Low"
                        ),
                        "Feedback": (
                            item["correct_class"] if item["feedback_received"] else "None"
                        ),
                    }
                )

            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.error("Failed to fetch history.")
    except Exception as e:
        st.error(f"Could not connect to the backend server. Is FastAPI running? {e}")
