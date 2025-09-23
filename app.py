# app.py (Streamlit + Vertex AI, with Gemini refinement for prompts)
import os
import datetime
import json
import streamlit as st

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel

# ---------------- CONFIG ----------------
PROJECT_ID = "drl-zenai-prod"
REGION = "us-central1"

# Load Google Cloud credentials from Streamlit secrets
creds_path = "/tmp/service_account.json"
service_account_info = dict(st.secrets["gcp_service_account"])
with open(creds_path, "w") as f:
    json.dump(service_account_info, f)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

# Init Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)

# Models
IMAGE_MODEL_NAME = "imagen-4.0-generate-001"
IMAGE_MODEL = ImageGenerationModel.from_pretrained(IMAGE_MODEL_NAME)

TEXT_MODEL_NAME = "gemini-2.0-flash"
TEXT_MODEL = GenerativeModel(TEXT_MODEL_NAME)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Image Generator", layout="wide")
st.title("🖼️ AI Image Generator (with Gemini Prompt Refinement)")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# ---------------- UI ----------------
raw_prompt = st.text_area("✨ Enter your prompt to generate an image:", height=120)

# Aspect ratio selector
aspect_ratio = st.selectbox(
    "📐 Choose Aspect Ratio",
    options=["1:1", "16:9", "9:16", "4:3", "3:2"],
    index=0
)

# Resolution selector (square sizes for Imagen)
resolution = st.selectbox(
    "🖼️ Choose Resolution",
    options=["512x512", "1024x1024", "2048x2048"],
    index=1
)

# Convert resolution string into tuple
width, height = map(int, resolution.split("x"))

if st.button("🚀 Generate Image"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        with st.spinner("Refining prompt with Gemini..."):
            try:
                refinement_prompt = f"""
You are an expert AI prompt engineer specialized in generating detailed, vivid, and creative image prompts.

Task:
- Take the user’s raw image request.
- Expand and refine it into a detailed, descriptive prompt suitable for an AI image generation model.
- Add missing details such as environment, lighting, style, perspective, mood, and realism level, while keeping true to the user’s intent.
- Do NOT change the subject of the request.
- Keep the language concise but expressive.
- Output only the final enhanced prompt, nothing else.

User’s raw prompt:
"{raw_prompt}"

Enhanced image generation prompt:
"""
                response = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = response.text.strip()

                st.info(f"🔮 Enhanced Prompt:\n\n{enhanced_prompt}")

            except Exception as e:
                st.error(f"⚠️ Gemini prompt refinement error: {e}")
                st.stop()

        with st.spinner("Generating image with Imagen 4..."):
            try:
                resp = IMAGE_MODEL.generate_images(
                    prompt=enhanced_prompt,
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,     # ✅ aspect ratio
                    size=f"{width}x{height}"       # ✅ resolution
                )

                if resp.images and hasattr(resp.images[0], "_image_bytes"):
                    img_bytes = resp.images[0]._image_bytes
                else:
                    st.error("❌ No image data returned from Imagen 4")
                    st.stop()

                # Save file locally (optional on Streamlit Cloud)
                output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
                os.makedirs(output_dir, exist_ok=True)
                filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                filepath = os.path.join(output_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(img_bytes)

                # Show image
                st.image(img_bytes, caption=filename, use_container_width=True)

                # Download button
                st.download_button(
                    "⬇️ Download Image",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/png"
                )

                # Store in session history
                st.session_state.generated_images.append({"filename": filename, "content": img_bytes})

            except Exception as e:
                st.error(f"⚠️ Image generation error: {e}")

# ---------------- HISTORY ----------------
if st.session_state.generated_images:
    st.subheader("📂 Past Generated Images")
    for i, img in enumerate(st.session_state.generated_images):
        with st.expander(f"Image {i+1}: {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
