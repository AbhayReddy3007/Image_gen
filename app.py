
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


style = st.selectbox("🎨 Choose Style", options=list(STYLE_DESCRIPTIONS.keys()), index=0)

raw_prompt = st.text_area("✨ Enter your prompt to generate an image:", height=120)

num_images = st.slider("🧾 Number of images", min_value=1, max_value=4, value=1)

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.

Your job:
- Transform the raw input into a compelling, professional, campaign-ready image prompt.
- Expand with persuasive details about:
  • Background and setting (modern, lifestyle, commercial, aspirational)
  • Lighting and atmosphere (studio lights, golden hour, cinematic)
  • Style (photorealistic, cinematic, product photography, lifestyle branding)
  • Perspective and composition (wide shot, close-up, dramatic angles)
  • Mood, tone, and branding suitability (premium, sleek, aspirational)

Rules:
- Stay faithful to the user’s idea but elevate it for ads, social media, or presentations.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined marketing image prompt:""",

    "Design": """You are a senior AI prompt engineer supporting a creative design team.

Your job:
- Expand raw input into a visually inspiring, design-oriented image prompt.
- Add imaginative details about:
  • Artistic styles (minimalist, abstract, futuristic, flat, 3D render, watercolor, digital illustration)
  • Color schemes, palettes, textures, and patterns
  • Composition and balance (symmetry, negative space, creative framing)
  • Lighting and atmosphere (soft glow, vibrant contrast, surreal shading)
  • Perspective (isometric, top-down, wide shot, close-up)

Rules:
- Keep fidelity to the idea but make it highly creative and visually unique.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined design image prompt:""",

    "General": """You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

Your job:
- Expand the user’s input into a detailed, clear prompt for an image generation model.
- Add missing details such as:
  • Background and setting
  • Lighting and mood
  • Style and realism level
  • Perspective and composition

Rules:
- Stay true to the user’s intent.
- Keep language concise, descriptive, and expressive.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:""",

    "DPEX": """You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and technology-focused image prompt.
- Expand with contextual details about:
  • Technology environments (server rooms, data centers, cloud systems, coding workspaces)
  • Digital elements (network diagrams, futuristic UIs, holograms, cybersecurity visuals)
  • People in IT roles (developers, engineers, admins, tech support, collaboration)
  • Tone (innovative, technical, futuristic, professional)
  • Composition (screens, servers, code on monitors, abstract digital patterns)
  • Lighting and effects (LED glow, cyberpunk tones, neon highlights, modern tech ambiance)

Rules:
- Ensure images are suitable for IT presentations, product demos, training, technical documentation, and digital transformation campaigns.
- Stay true to the user’s intent but emphasize a technological and innovative look.
- Output only the final refined image prompt.

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

                # Force resize to 2048x2048
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
                    filename = f"{department.lower()}_{style.lower()}_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}_2048.png"
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
    for i, img in enumerate(reversed(st.session_state.generated_images[-20:])):  # last 20
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
