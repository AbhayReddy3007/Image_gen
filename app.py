# app.py (Streamlit + Vertex AI, with Gemini refinement for prompts + explicit resolution control)
import os
import datetime
import json
import io
import streamlit as st
from PIL import Image

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

# Resolution selector (explicit width × height)
resolution = st.selectbox(
    "🖼️ Choose Resolution",
    options=["512x512", "1024x1024", "1920x1080", "1080x1920", "2048x2048"],
    index=1
)
target_width, target_height = map(int, resolution.split("x"))

# Number of images
num_images = st.slider("🧾 Number of images", min_value=1, max_value=4, value=1)

# ---------------- Helpers ----------------
def safe_get_enhanced_text(resp):
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

def get_image_bytes_from_genobj(gen_obj):
    if isinstance(gen_obj, (bytes, bytearray)):
        return bytes(gen_obj)
    for attr in ["image_bytes", "_image_bytes"]:
        if hasattr(gen_obj, attr):
            return getattr(gen_obj, attr)
    if hasattr(gen_obj, "image") and gen_obj.image:
        for attr in ["image_bytes", "_image_bytes"]:
            if hasattr(gen_obj.image, attr):
                return getattr(gen_obj.image, attr)
    return None

def resize_to_resolution(img_bytes, target_w, target_h):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    resized = img.resize((target_w, target_h), Image.LANCZOS)
    out = io.BytesIO()
    resized.save(out, format="PNG")
    return out.getvalue()

# ---------------- Generation flow ----------------
if st.button("🚀 Generate Image"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        # 1) refine prompt with Gemini
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
\"{raw_prompt}\"

Enhanced image generation prompt:
"""
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                

            except Exception as e:
                st.error(f"⚠️ Gemini prompt refinement error: {e}")
                st.stop()

        # 2) generate images
        with st.spinner("Generating image(s) with Imagen..."):
            try:
                resp = IMAGE_MODEL.generate_images(
                    prompt=enhanced_prompt,
                    number_of_images=num_images,
                    aspect_ratio=aspect_ratio,
                )
            except Exception as e:
                st.error(f"⚠️ Image generation error: {e}")
                st.stop()

            generated_raws = []
            # Try to extract each image
            for i in range(num_images):
                gen_obj = None
                try:
                    gen_obj = resp.images[i]
                except Exception:
                    try:
                        gen_obj = resp[i]
                    except Exception:
                        pass

                if gen_obj is None:
                    continue

                img_bytes = get_image_bytes_from_genobj(gen_obj)
                if not img_bytes:
                    continue

                # Force resize to chosen resolution
                try:
                    img_bytes = resize_to_resolution(img_bytes, target_width, target_height)
                except Exception:
                    pass

                generated_raws.append(img_bytes)

            # 3) Show images in a grid and save
            if generated_raws:
                cols = st.columns(len(generated_raws))
                for idx, img_bytes in enumerate(generated_raws):
                    col = cols[idx]
                    filename = f"image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                    output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
                    os.makedirs(output_dir, exist_ok=True)
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    with col:
                        st.image(img_bytes, caption=filename, use_column_width=True)
                        st.download_button(
                            "⬇️ Download",
                            data=img_bytes,
                            file_name=filename,
                            mime="image/png",
                            key=f"dl_{filename}"
                        )
                    st.session_state.generated_images.append({"filename": filename, "content": img_bytes})
            else:
                st.error("❌ No images produced by the model.")

# ---------------- HISTORY ----------------
if st.session_state.generated_images:
    st.subheader("📂 Past Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-20:])):  # recent 20
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
