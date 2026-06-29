import requests
import streamlit as st

try:
    API_URL = st.secrets["API_URL"]
except Exception:
    API_URL = "http://127.0.0.1:8000/api/v1"


def render_upload_section():
    st.header("Upload CT Scan")

    uploaded_file = st.file_uploader(
        "Drag and drop a kidney CT scan image here", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.session_state.uploaded_image = uploaded_file.getvalue()
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    response = requests.post(f"{API_URL}/predict", files=files)

                    if response.status_code == 200:
                        st.session_state.prediction_result = response.json()
                        st.session_state.uploaded_image = uploaded_file.getvalue()
                        st.session_state.pdf_report_bytes = None
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        "Failed to connect to the FastAPI backend. "
                        "Is it running on http://127.0.0.1:8000?"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
