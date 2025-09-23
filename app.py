# app.py (Streamlit + Vertex AI) — fixed: no `size=` arg on generate_images, adds robust upscaling/resizing
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

# Resolution selector (interpreted as 'longest side' target)
resolution_choice = st.selectbox(
    "🖼️ Target longest side (px)",
    options=["512", "1024", "2048"],
    index=1
)
target_side = int(resolution_choice)

# Number of images
num_images = st.slider("🧾 Number of images", min_value=1, max_value=4, value=1)

generate_clicked = st.button("🚀 Generate Image")

# ---------------- Helpers ----------------
def safe_get_enhanced_text(resp):
    """Try a few common fields for the text returned by the GenerativeModel."""
    if resp is None:
        return None
    # direct .text
    if hasattr(resp, "text") and resp.text:
        return resp.text
    # .generations (list) -> .text
    if hasattr(resp, "generations") and resp.generations:
        try:
            gen = resp.generations[0]
            if hasattr(gen, "text") and gen.text:
                return gen.text
        except Exception:
            pass
    # .candidates (list)
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            cand = resp.candidates[0]
            # candidate content might be .content or .text or nested
            if hasattr(cand, "content") and isinstance(cand.content, str):
                return cand.content
            if hasattr(cand, "text") and cand.text:
                return cand.text
        except Exception:
            pass
    # fallback to str(resp)
    return str(resp)

def get_image_bytes_from_genobj(gen_obj):
    """Try multiple locations to extract raw image bytes from a generated-image object."""
    if gen_obj is None:
        return None

    # if the object is already bytes
    if isinstance(gen_obj, (bytes, bytearray)):
        return bytes(gen_obj)

    checks = [
        lambda g: getattr(g, "image_bytes", None),
        lambda g: getattr(g, "_image_bytes", None),
        lambda g: getattr(g, "image", None) and getattr(g.image, "image_bytes", None),
        lambda g: getattr(g, "image", None) and getattr(g.image, "_image_bytes", None),
        lambda g: getattr(g, "image", None) and getattr(g.image, "image", None) and getattr(g.image.image, "_image_bytes", None),
        lambda g: getattr(g, "image", None) and getattr(g.image, "image", None) and getattr(g.image.image, "image_bytes", None),
    ]
    for fn in checks:
        try:
            val = fn(gen_obj)
            if val:
                return bytes(val)
        except Exception:
            continue
    # sometimes the object is iterable with .images
    try:
        if hasattr(gen_obj, "images") and gen_obj.images:
            first = gen_obj.images[0]
            return get_image_bytes_from_genobj(first)
    except Exception:
        pass
    return None

def resize_preserve_aspect(img_bytes, aspect_ratio_str, target_long_side):
    """Resize (or upscale) raw image bytes to target_long_side while preserving given aspect ratio.
       target_long_side means: the largest dimension should be target_long_side px.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    ratio_w, ratio_h = map(int, aspect_ratio_str.split(":"))
    # compute target dims with target_long_side as longest side
    if ratio_w >= ratio_h:
        target_w = target_long_side
        target_h = max(1, round(target_long_side * ratio_h / ratio_w))
    else:
        target_h = target_long_side
        target_w = max(1, round(target_long_side * ratio_w / ratio_h))
    # resize with high-quality resampling
    resized = img.resize((target_w, target_h), Image.LANCZOS)
    out = io.BytesIO()
    # Keep PNG to preserve alpha if any
    resized.save(out, format="PNG")
    return out.getvalue()

# ---------------- Generation flow ----------------
if generate_clicked:
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
                st.info(f"🔮 Enhanced Prompt:\n\n{enhanced_prompt}")

            except Exception as e:
                st.error(f"⚠️ Gemini prompt refinement error: {e}")
                st.stop()

        # 2) generate images (no 'size' arg)
        with st.spinner("Generating image(s) with Imagen..."):
            try:
                resp = IMAGE_MODEL.generate_images(
                    prompt=enhanced_prompt,
                    number_of_images=num_images,
                    aspect_ratio=aspect_ratio,
                )
            except TypeError as e:
                # defensive: if API signature changed, give helpful error
                st.error(f"⚠️ Image generation call failed: {e}")
                st.stop()
            except Exception as e:
                st.error(f"⚠️ Image generation error: {e}")
                st.stop()

            # resp may be list-like or have .images — attempt to iterate
            generated_raws = []  # list of bytes
            for i in range(num_images):
                gen_obj = None
                # try common resp shapes
                try:
                    # resp could be indexable
                    gen_obj = resp[i]
                except Exception:
                    try:
                        if hasattr(resp, "images"):
                            gen_obj = resp.images[i]
                    except Exception:
                        # try iterating
                        try:
                            gen_obj = list(resp)[i]
                        except Exception:
                            gen_obj = None

                if gen_obj is None:
                    st.warning(f"Could not extract image object for result #{i+1}")
                    continue

                # get bytes
                img_bytes = get_image_bytes_from_genobj(gen_obj)
                # if user requested upscale to 2048, try model.upscale_image if available
                if target_side >= 2048:
                    up_success = False
                    if hasattr(IMAGE_MODEL, "upscale_image"):
                        try:
                            # pass the generated-image object if the SDK wants it (per docs)
                            up_resp = IMAGE_MODEL.upscale_image(image=gen_obj, new_size=2048)
                            # try to extract bytes from up_resp
                            up_bytes = get_image_bytes_from_genobj(up_resp)
                            if up_bytes:
                                img_bytes = up_bytes
                                up_success = True
                        except Exception:
                            up_success = False
                    if not up_success:
                        # fallback: client-side upscale (PIL). Less high-quality, but workable.
                        if img_bytes:
                            try:
                                img_bytes = resize_preserve_aspect(img_bytes, aspect_ratio, target_side)
                            except Exception:
                                # if resizing fails, keep original bytes
                                pass
                else:
                    # downscale or adjust to target_side preserving aspect ratio:
                    if img_bytes:
                        try:
                            img_bytes = resize_preserve_aspect(img_bytes, aspect_ratio, target_side)
                        except Exception:
                            # keep original if PIL fails
                            pass

                if not img_bytes:
                    st.warning(f"Unable to get bytes for image #{i+1}")
                    continue

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
                    # add to session history
                    st.session_state.generated_images.append({"filename": filename, "content": img_bytes})
            else:
                st.error("❌ No images produced by the model.")

# ---------------- HISTORY ----------------
if st.session_state.generated_images:
    st.subheader("📂 Past Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-20:])):  # show recent 20
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
