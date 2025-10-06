import os
import re
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
if "regenerated" not in st.session_state:
    st.session_state.regenerated = {}  # mapping safe_key -> list of {"filename","content"}
# regen_error_<safe_key> and regen_done_<safe_key> will be set dynamically as needed

# ---------------- UI ----------------
department = st.selectbox(
    "🏢 Select Department",
    options=["Marketing", "Design", "General", "DPEX", "HR", "Business", "None"],
    index=2
)

STYLE_DESCRIPTIONS = {
    "None": "No special styling — keep the image natural, faithful to the user’s idea.",
    "Smart": "A clean, balanced, and polished look. Professional yet neutral, visually appealing without strong artistic bias.",
    "Cinematic": "Film-style composition with professional lighting. Wide dynamic range, dramatic highlights, storytelling feel.",
    "Cinematic Concept": "Epic and dramatic like a movie concept frame. Wide cinematic perspective, moody atmosphere, detailed world-building.",
    "Creative": "Playful, imaginative, and experimental. Bold artistic choices, unexpected elements, and expressive color use.",
    "Bokeh": "Photography style with shallow depth of field. Subject in sharp focus with soft, dreamy, blurred backgrounds.",
    "Macro": "Extreme close-up photography. High detail, textures visible, shallow focus highlighting minute features.",
    "Illustration": "Hand-drawn or digitally illustrated style. Clear outlines, stylized shading, expressive and artistic.",
    "3D Render": "Photorealistic or stylized CGI. Crisp geometry, depth, shadows, and reflective surfaces with realistic rendering.",
    "Fashion": "High-end editorial photography. Stylish, glamorous poses, bold makeup, controlled lighting, and modern aesthetic.",
    "Minimalist": "Simple and uncluttered. Few elements, large negative space, flat or muted color palette, clean composition.",
    "Moody": "Dark, atmospheric, and emotional. Strong shadows, high contrast, deep tones, cinematic ambiance.",
    "Portrait": "Focus on the subject. Natural skin tones, shallow depth of field, close-up or waist-up framing, studio or natural lighting.",
    "Stock Photo": "Professional, commercial-quality photo. Neutral subject matter, polished composition, business-friendly aesthetic.",
    "Vibrant": "Bold, saturated colors. High contrast, energetic mood, eye-catching and lively presentation.",
    "Pop Art": "Comic-book and pop-art inspired. Bold outlines, halftone patterns, flat vivid colors, high contrast, playful tone.",
    "Vector": "Flat vector graphics. Smooth shapes, sharp edges, solid fills, and clean scalable style like logos or icons.",

    "Watercolor": "Soft, fluid strokes with delicate blending and washed-out textures. Artistic and dreamy.",
    "Oil Painting": "Rich, textured brushstrokes. Classic fine art look with deep color blending.",
    "Charcoal": "Rough, sketchy textures with dark shading. Artistic, raw, dramatic effect.",
    "Line Art": "Minimal monochrome outlines with clean, bold strokes. No shading, focus on form.",

    "Anime": "Japanese animation style with vibrant colors, clean outlines, expressive features, and stylized proportions.",
    "Cartoon": "Playful, exaggerated features, simplified shapes, bold outlines, and bright colors.",
    "Pixel Art": "Retro digital art style. Small, pixel-based visuals resembling old-school video games.",

    "Fantasy Art": "Epic fantasy scenes. Magical elements, mythical creatures, enchanted landscapes.",
    "Surreal": "Dreamlike, bizarre imagery. Juxtaposes unexpected elements, bending reality.",
    "Concept Art": "Imaginative, detailed artwork for games or films. Often moody and cinematic.",

    "Cyberpunk": "Futuristic neon city vibes. High contrast, glowing lights, dark tones, sci-fi feel.",
    "Steampunk": "Retro-futuristic style with gears, brass, Victorian aesthetics, and industrial design.",
    "Neon Glow": "Bright neon outlines and glowing highlights. Futuristic, nightlife aesthetic.",
    "Low Poly": "Simplified 3D style using flat geometric shapes and polygons.",
    "Isometric": "3D look with isometric perspective. Often used for architecture, games, and diagrams.",

    "Vintage": "Old-school, retro tones. Faded colors, film grain, sepia, or retro print feel.",
    "Graffiti": "Urban street art style with bold colors, spray paint textures, and rebellious tone."
}

# ---------------- Department-aware Style Suggestions ----------------
DEPARTMENT_STYLE_MAP = {
    "Marketing": ["Fashion", "Vibrant", "Stock Photo", "Cinematic", "Minimalist"],
    "Design": ["Vector", "Creative", "Pop Art", "Illustration", "3D Render"],
    "General": ["Smart", "Cinematic", "Portrait", "Stock Photo"],
    "DPEX": ["Moody", "Cinematic Concept", "3D Render", "Minimalist", "Cyberpunk"]
}

def get_styles_for_department(dept):
    base_styles = DEPARTMENT_STYLE_MAP.get(dept, [])
    all_styles = ["None"] + base_styles + [s for s in STYLE_DESCRIPTIONS.keys() if s not in base_styles and s != "None"]
    return all_styles

styles_for_dept = get_styles_for_department(department)

style = st.selectbox(
    "🎨 Choose Style",
    options=styles_for_dept,
    index=0
)

raw_prompt = st.text_area("Enter your prompt to generate an image:", height=120)

num_images = 1

# ---------------- Prompt Templates ----------------
PROMPT_TEMPLATES = {
    "Marketing": """
You are a senior AI prompt engineer creating polished prompts for marketing and advertising visuals.

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

Refined marketing image prompt:
""",

    "Design": """
You are a senior AI prompt engineer supporting a creative design team.

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

Refined design image prompt:
""",

    "General": """
You are an expert AI prompt engineer specialized in creating vivid and descriptive image prompts.

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

Refined general image prompt:
""",
    "None": """
Dont make any changes in the user's prompt.Follow it as it is
User’s raw prompt:
"{USER_PROMPT}"

Refined general image prompt:
""",

    "DPEX": """
You are a senior AI prompt engineer creating refined prompts for IT and technology-related visuals.

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

Refined DPEX image prompt:
""",

    "HR": """
You are a senior AI prompt engineer creating refined prompts for human resources and workplace-related visuals.

Your job:
- Transform the raw input into a detailed, professional, and HR-focused image prompt.
- Expand with contextual details about:
  • Workplace settings (modern office, meeting rooms, open workspaces, onboarding sessions)
  • People interactions (interviews, teamwork, training, collaboration, diversity and inclusion)
  • Themes (employee engagement, professional growth, recruitment, performance evaluation)
  • Composition (groups in discussion, managers mentoring, collaborative workshops)
  • Lighting and tone (bright, welcoming, professional, inclusive)

Rules:
- Ensure images are suitable for HR presentations, recruitment campaigns, internal training, or employee engagement material.
- Stay true to the user’s intent but emphasize people, culture, and workplace positivity.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined HR image prompt:
""",

    "Business": """
You are a senior AI prompt engineer creating refined prompts for business and corporate visuals.

Your job:
- Transform the raw input into a detailed, professional, and business-oriented image prompt.
- Expand with contextual details about:
  • Corporate settings (boardrooms, skyscrapers, modern offices, networking events)
  • Business activities (presentations, negotiations, brainstorming sessions, teamwork)
  • People (executives, entrepreneurs, consultants, diverse teams, global collaboration)
  • Tone (professional, ambitious, strategic, innovative)
  • Composition (formal meetings, handshake deals, conference tables, city skyline backgrounds)
  • Lighting and atmosphere (clean, modern, premium, professional)

Rules:
- Ensure images are suitable for corporate branding, investor decks, strategy sessions, or professional reports.
- Stay true to the user’s intent but emphasize professionalism, ambition, and success.
- Output only the final refined image prompt.

User’s raw prompt:
"{USER_PROMPT}"

Refined business image prompt:
"""
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

def _sanitize_key(s: str):
    # make a safe key for session_state from filename or label
    return re.sub(r'[^0-9a-zA-Z_-]+', '_', s)

# ---------------- Regeneration callback ----------------
def regenerate_callback(safe_key, dept, style, new_prompt_key):
    """
    This callback runs when a regeneration button is clicked.
    It performs Gemini refinement + Imagen generation and stores results in session_state.
    """
    try:
        new_prompt = st.session_state.get(new_prompt_key, "")
        if not new_prompt or not new_prompt.strip():
            st.session_state[f"regen_error_{safe_key}"] = "Prompt is empty."
            return

        # 1) refine prompt with Gemini
        refinement_prompt = PROMPT_TEMPLATES[dept].replace("{USER_PROMPT}", new_prompt)
        if style != "None":
            refinement_prompt += f"\n\nApply the visual style: {STYLE_DESCRIPTIONS[style]}"

        text_resp = TEXT_MODEL.generate_content(refinement_prompt)
        enhanced_prompt = safe_get_enhanced_text(text_resp).strip()

        # 2) call Imagen
        resp = IMAGE_MODEL.generate_images(prompt=enhanced_prompt, number_of_images=1)

        # robustly extract image object
        gen_obj = None
        try:
            gen_obj = resp.images[0]
        except Exception:
            try:
                gen_obj = resp[0]
            except Exception:
                gen_obj = None

        if not gen_obj:
            st.session_state[f"regen_error_{safe_key}"] = "Model response contained no image."
            return

        img_bytes = get_image_bytes_from_genobj(gen_obj)
        if not img_bytes:
            st.session_state[f"regen_error_{safe_key}"] = "Failed to extract image bytes from model response."
            return

        try:
            img_bytes = resize_to_2048(img_bytes)
        except Exception:
            pass

        # save generated file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        regen_fname = f"{safe_key}_regenerated_{timestamp}.png"
        output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, regen_fname)
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        # store in session_state so UI can show it after rerun
        st.session_state.generated_images.append({"filename": regen_fname, "content": img_bytes})
        st.session_state.regenerated.setdefault(safe_key, []).append({"filename": regen_fname, "content": img_bytes})

        # clear previous error and mark done
        st.session_state.pop(f"regen_error_{safe_key}", None)
        st.session_state[f"regen_done_{safe_key}"] = True

    except Exception as e:
        st.session_state[f"regen_error_{safe_key}"] = str(e)

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
                    safe_key = _sanitize_key(filename)

                    output_dir = os.path.join(os.path.dirname(__file__), "generated_images")
                    os.makedirs(output_dir, exist_ok=True)
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    with col:
                        st.image(img_bytes, caption=filename, use_column_width=True)

                        # Download button
                        st.download_button(
                            "⬇️ Download",
                            data=img_bytes,
                            file_name=filename,
                            mime="image/png",
                            key=f"dl_{safe_key}"
                        )

                        # Prepare default new_prompt session key so the callback can read it
                        new_prompt_key = f"new_prompt_{safe_key}"
                        if new_prompt_key not in st.session_state:
                            st.session_state[new_prompt_key] = raw_prompt

                        # Change prompt and regenerate (uses callback)
                        with st.expander(f"✏️ Change Prompt & Regenerate ({idx+1})"):
                            st.text_area(
                                f"Enter a new prompt for {filename}",
                                key=new_prompt_key,
                                height=120
                            )
                            st.button(
                                f"🔄 Regenerate {idx+1}",
                                key=f"regen_btn_{safe_key}",
                                on_click=regenerate_callback,
                                args=(safe_key, department, style, new_prompt_key)
                            )

                        # Show regeneration errors if any
                        if st.session_state.get(f"regen_error_{safe_key}"):
                            st.error(st.session_state.get(f"regen_error_{safe_key}"))

                        # Show regenerated images (if any)
                        regenerated_list = st.session_state.regenerated.get(safe_key, [])
                        if regenerated_list:
                            st.markdown("**Regenerated versions:**")
                            for r in regenerated_list:
                                st.image(r["content"], caption=f"Regenerated: {r['filename']}", use_column_width=True)

                    # persist the original into history
                    st.session_state.generated_images.append({"filename": filename, "content": img_bytes})
            else:
                st.error("❌ No images produced by the model.")

# ---------------- HISTORY ----------------
if st.session_state.generated_images:
    st.subheader("📂 Past Generated Images")
    for i, img in enumerate(reversed(st.session_state.generated_images[-40:])):  # last 40
        with st.expander(f"{i+1}. {img['filename']}"):
            st.image(img["content"], caption=img["filename"], use_container_width=True)
            st.download_button(
                "⬇️ Download Again",
                data=img["content"],
                file_name=img["filename"],
                mime="image/png",
                key=f"download_img_{i}"
            )
