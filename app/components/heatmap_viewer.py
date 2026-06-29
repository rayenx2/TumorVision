import base64

import streamlit as st


def render_heatmap_viewer():
    if "prediction_result" not in st.session_state or "uploaded_image" not in st.session_state:
        return

    st.subheader("Image Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original CT Scan**")
        st.image(st.session_state.uploaded_image, use_container_width=True)

    with col2:
        st.markdown("**Grad-CAM Heatmap**")
        gradcam_base64 = st.session_state.prediction_result.get("gradcam_base64", "")
        if gradcam_base64:
            # The base64 string from FastAPI might have the 'data:image/jpeg;base64,' prefix
            if "," in gradcam_base64:
                gradcam_base64 = gradcam_base64.split(",")[1]
            try:
                image_bytes = base64.b64decode(gradcam_base64)
                st.image(image_bytes, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to render heatmap image: {e}")
        else:
            st.info("No heatmap available for this prediction.")
