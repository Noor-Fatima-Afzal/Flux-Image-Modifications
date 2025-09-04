import os
import torch
import streamlit as st
from diffusers import FluxPipeline
from huggingface_hub import login
from PIL import Image
import io

# ---- Sidebar for Hugging Face Token ----
st.sidebar.title("Authentication")
token = st.sidebar.text_input(
    "Enter your Hugging Face token:",
    type="password",
    value=os.getenv("HF_TOKEN", "")
)

if token:
    try:
        login(token)
        st.sidebar.success("Authenticated successfully ✅")
    except Exception as e:
        st.sidebar.error(f"Login failed: {e}")
        st.stop()
else:
    st.sidebar.warning("Please provide your Hugging Face token.")
    st.stop()

# ---- Load Model ----
@st.cache_resource
def load_pipeline():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    return pipe

pipe = load_pipeline()

# ---- App UI ----
st.title("🖼️ FLUX.1-dev Image-to-Image Editor")
st.write("Upload an image and modify it with your text prompt.")

# Inputs
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
prompt = st.text_area("Enter your prompt", "Make it look like a painting in Van Gogh's style")
strength = st.slider("Strength (0.1 = subtle, 1.0 = strong)", 0.1, 1.0, 0.7, 0.05)
guidance = st.slider("Guidance scale", 1.0, 10.0, 4.0, 0.5)
steps = st.slider("Number of inference steps", 10, 100, 40, 5)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate"):
        with st.spinner("Generating... please wait"):
            init_image = Image.open(uploaded_file).convert("RGB")

            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps
            )

            output_img = result.images[0]

            # Show output
            st.image(output_img, caption="Modified Image", use_column_width=True)

            # Download
            buf = io.BytesIO()
            output_img.save(buf, format="PNG")
            st.download_button(
                label="Download Modified Image",
                data=buf.getvalue(),
                file_name="flux_modified.png",
                mime="image/png"
            )
