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
st.title("🖼️ AI Image Generator")

# ---------------- STATE ----------------
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

# ---------------- UI ----------------
department = st.selectbox(
    "🏢 Select Department",
    options=["Marketing", "Design", "General", "DPEX"],
    index=2
)

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic Concept": "Epic and dramatic like a movie concept frame. Wide cinematic perspective, moody atmosphere, detailed world-building.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Bokeh": "Photography style with shallow depth of field. Subject in sharp focus with soft, dreamy, blurred backgrounds.",
    "Macro": "Extreme close-up photography. High detail, textures visible, shallow focus highlighting minute features.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Fashion": "High-end editorial photography. Stylish, glamorous poses, bold makeup, controlled lighting, and modern aesthetic.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Moody": "Dark, atmospheric, and emotional. Strong shadows, high contrast, deep tones, cinematic ambiance.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Sketch - Color": "Colored pencil or ink sketch. Visible strokes, hand-drawn imperfections, vibrant but artistic look.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Ray Traced": "CGI realism with ray-traced reflections, shadows, and refractions. Polished, highly realistic rendering.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Sketch - Black & White": "Hand-drawn monochrome sketch with pencil or ink. Strong lines, shading for depth, artistic rawness.",
    "Pop Art": "Comic-book and pop-art inspired. Bold outlines, halftone patterns, flat vivid colors, high contrast, playful tone.",
    "Vector": "Flat vector graphics. Smooth shapes, sharp edges, solid fills, and clean scalable style like logos or icons."
}

# ---------------- Department-aware Style Suggestions ----------------
DEPARTMENT_STYLE_MAP = {
    "Marketing": ["Fashion", "Vibrant", "Stock Photo", "Cinematic", "Minimalist"],
    "Design": ["Vector", "Creative", "Pop Art", "Illustration", "3D Render"],
    "General": ["Smart", "Cinematic", "Portrait", "Stock Photo"],
    "DPEX": ["Moody", "Cinematic Concept", "3D Render", "Ray Traced", "Minimalist"]
}

def get_styles_for_department(dept):
    base_styles = DEPARTMENT_STYLE_MAP.get(dept, [])
    all_styles = ["None"] + base_styles + [s for s in STYLE_DESCRIPTIONS.keys() if s not in base_styles and s != "None"]
    return all_styles

styles_for_dept = get_styles_for_department(department)

style = st.selectbox(
    "🎨 Choose Style",
    options=styles_for_dept,
    index=0  # defaults to "None"
)

# ---------------- Mode Selection ----------------
mode = st.radio(
    "✨ Generation Mode",
    ["Text-to-Image", "Image-to-Image"],
    index=0,
    horizontal=True
)

uploaded_image = None
mask_image = None
if mode == "Image-to-Image":
    uploaded_image = st.file_uploader("📤 Upload an image to modify", type=["png", "jpg", "jpeg"])
    mask_image = st.file_uploader("🎭 (Optional) Upload a mask for inpainting", type=["png"])

    if uploaded_image:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image, caption="🖼️ Base Image", use_container_width=True)
        if mask_image:
            with col2:
                st.image(mask_image, caption="🎭 Mask Image", use_container_width=True)

raw_prompt = st.text_area("✨ Enter your prompt:", height=120)
num_images = st.slider("🧾 Number of images", min_value=1, max_value=4, value=1)

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.

User’s raw prompt:
"{USER_PROMPT}"

Refined marketing image prompt:""",
    "Design": """You are a senior AI prompt engineer supporting a creative design team.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:""",
    "General": """You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:""",
    "DPEX": """You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

User’s raw prompt:
"{USER_PROMPT}"

Refined DPEX image prompt:"""
}

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

def resize_to_2048(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    resized = img.resize((2048, 2048), Image.LANCZOS)
    out = io.BytesIO()
    resized.save(out, format="PNG")
    return out.getvalue()

# ---------------- Generation flow ----------------
if st.button("🚀 Generate Image"):
    if not raw_prompt.strip():
        st.warning("Please enter a prompt!")
    elif mode == "Image-to-Image" and uploaded_image is None:
        st.warning("Please upload an image for Image-to-Image mode!")
    else:
        # 1) refine prompt with Gemini
        with st.spinner("Refining prompt with Gemini..."):
            try:
                refinement_prompt = PROMPT_TEMPLATES[department].replace("{USER_PROMPT}", raw_prompt)
                if style != "None":
                    refinement_prompt += f"\n\nApply the visual style: {STYLE_DESCRIPTIONS[style]}"
                text_resp = TEXT_MODEL.generate_content(refinement_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Prompt ({department} / {style}):\n\n{enhanced_prompt}")
            except Exception as e:
                st.error(f"⚠️ Gemini prompt refinement error: {e}")
                st.stop()

        # 2) generate images
        with st.spinner("Generating image(s) with Imagen..."):
            try:
                if mode == "Image-to-Image" and uploaded_image is not None:
                    base_img = Image.open(uploaded_image).convert("RGBA")
                    buf = io.BytesIO()
                    base_img.save(buf, format="PNG")
                    buf.seek(0)

                    mask_bytes = mask_image.read() if mask_image else None

                    resp = IMAGE_MODEL.generate_images(
                        prompt=enhanced_prompt,
                        number_of_images=num_images,
                        image=buf.getvalue(),
                        mask=mask_bytes
                    )
                else:
                    resp = IMAGE_MODEL.generate_images(
                        prompt=enhanced_prompt,
                        number_of_images=num_images,
                    )
            except Exception as e:
                st.error(f"⚠️ Image generation error: {e}")
                st.stop()

            generated_raws = []
            for i in range(num_images):
                gen_obj = None
                try:
                    gen_obj = resp.images[i]
                except Exception:
                    try:
                        gen_obj = resp[i]
                    except Exception:
                        pass

                if not gen_obj:
                    continue

                img_bytes = get_image_bytes_from_genobj(gen_obj)
                if not img_bytes:
                    continue

                try:
                    img_bytes = resize_to_2048(img_bytes)
                except Exception:
                    pass

                generated_raws.append(img_bytes)

            # 3) Show images
            if generated_raws:
                cols = st.columns(len(generated_raws))
                for idx, img_bytes in enumerate(generated_raws):
                    col = cols[idx]
                    filename = f"{department.lower()}_{style.lower()}_{mode.lower()}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}_2048.png"
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
    for i, img in enumerate(reversed(st.session_state.generated_images[-20:])):
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
