import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/api/v1")


def render_result_card():
    if "prediction_result" not in st.session_state:
        return

    result = st.session_state.prediction_result

    st.subheader("Prediction Results")

    # Uncertainty Warning
    if result.get("is_uncertain", False):
        st.warning(
            f"⚠️ **High Uncertainty Detected** "
            f"(Score: {result.get('uncertainty_score', 0):.2f})."
            "This prediction requires radiologist review."
        )

    # Main prediction
    predicted_class = result.get("predicted_class", "Unknown")
    confidence = result.get("confidence", 0) * 100

    st.markdown(f"### Predicted Class: **{predicted_class}** " f"({confidence:.1f}% confidence)")

    # Confidence Bar Chart
    probabilities = result.get("probabilities", [])
    if probabilities:
        df = pd.DataFrame(probabilities)
        df["probability_pct"] = df["probability"] * 100

        fig = px.bar(
            df,
            x="probability_pct",
            y="class_name",
            orientation="h",
            labels={"probability_pct": "Confidence (%)", "class_name": "Class"},
            color="class_name",
            color_discrete_map={
                "Normal": "#2ECC71",
                "Cyst": "#3498DB",
                "Stone": "#F1C40F",
                "Tumor": "#E74C3C",
            },
        )
        fig.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Report Download
    st.markdown("---")
    st.markdown("**PDF Report:**")

    if "pdf_report_bytes" not in st.session_state:
        st.session_state.pdf_report_bytes = None

    if st.session_state.pdf_report_bytes is None:
        if st.button("Generate PDF Report", type="secondary"):
            with st.spinner("Generating PDF report..."):
                try:
                    # Trigger report generation from existing prediction
                    prediction_id = result.get("prediction_id")
                    image_bytes = st.session_state.uploaded_image
                    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}

                    response = requests.post(
                        f"{API_URL}/predict/{prediction_id}/report", files=files
                    )

                    if response.status_code == 200:
                        pred_data = response.json()
                        task_id = pred_data.get("task_id")

                        if task_id:
                            max_retries = 120  # Increased from 30 to 120 (4 minutes max wait)
                            retries = 0
                            report_ready = False

                            while retries < max_retries and not report_ready:
                                try:
                                    poll_resp = requests.get(f"{API_URL}/predict/report/{task_id}")
                                    if poll_resp.status_code == 200:
                                        if (
                                            poll_resp.headers.get("Content-Type")
                                            == "application/pdf"
                                        ):
                                            st.session_state.pdf_report_bytes = poll_resp.content
                                            report_ready = True
                                            st.rerun()
                                            break
                                        else:
                                            status_data = poll_resp.json()
                                            if status_data.get("status") == "failed":
                                                st.error("Report generation failed.")
                                                break
                                    time.sleep(2)
                                    retries += 1
                                except Exception:
                                    time.sleep(2)
                                    retries += 1

                            if not report_ready and retries >= max_retries:
                                st.warning(
                                    "Report generation is taking longer than expected."
                                    "Please try again later."
                                )
                        else:
                            st.error("Failed to get task ID for report generation.")
                    else:
                        st.error("Failed to initiate report generation.")
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    else:
        st.success("Report generated successfully!")
        st.download_button(
            label="📥 Download PDF Report",
            data=st.session_state.pdf_report_bytes,
            file_name=f"kidney_report_{result.get('prediction_id', 'report')}.pdf",
            mime="application/pdf",
            type="primary",
        )
