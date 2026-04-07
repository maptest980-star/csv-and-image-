import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from PIL import ImageFont
import io
import base64

st.set_page_config(layout="wide")
st.title("Image Annotation Viewer")

# Initialize session state for selected masks
if 'selected_masks' not in st.session_state:
    st.session_state.selected_masks = []

# Upload section
col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
with col2:
    annotation_file = st.file_uploader("Upload CSV", type=["csv"])

def display_high_quality(pil_image):
    """Render image at full quality using base64 PNG via HTML — bypasses Streamlit's JPEG compression."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG", optimize=False, compress_level=1)
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(
        f'<img src="data:image/png;base64,{img_b64}" style="width:100%; image-rendering: high-quality;" />',
        unsafe_allow_html=True,
    )

def draw_label(draw, x, y, label_text, font, is_selected):
    """Draw mask ID label with a solid background rectangle for visibility."""
    # Measure text size
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x, pad_y = 5, 3
    rect_x0 = x
    rect_y0 = y - text_h - pad_y * 2
    # If label would go above the image, place it inside the box top instead
    if rect_y0 < 0:
        rect_y0 = y
    rect_x1 = x + text_w + pad_x * 2
    rect_y1 = rect_y0 + text_h + pad_y * 2

    bg_color   = (139, 0, 0)   if is_selected else (0, 80, 200)
    text_color = (255, 255, 255)

    draw.rectangle([(rect_x0, rect_y0), (rect_x1, rect_y1)], fill=bg_color)
    draw.text((rect_x0 + pad_x, rect_y0 + pad_y), label_text, font=font, fill=text_color)

if image_file and annotation_file:
    # Load image
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    img_array = np.array(image)
    height, width, _ = img_array.shape
    
    # Load and process CSV
    try:
        df = pd.read_csv(annotation_file)

        # -------- CLEAN CSV DATA --------
        num_cols = [
            'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height',
            'confidence', 'detic_confidence',
            'char_count', 'numeric_count', 'ocr_confidence', 'yolo_confidence'
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        bool_cols = ['has_text']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().map({'TRUE': True, 'FALSE': False}).fillna(False)

        df['object_id'] = df['object_id'].astype(str)

        required_cols = ['object_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col != 'object_id' else ''

        rotation_cols = [
            'confidence_0deg', 'confidence_90deg',
            'confidence_180deg', 'confidence_270deg'
        ]
        if all(col in df.columns for col in rotation_cols):
            df['confidence'] = df[rotation_cols].max(axis=1)
            text_cols = [
                'extracted_text_0deg', 'extracted_text_90deg',
                'extracted_text_180deg', 'extracted_text_270deg'
            ]
            df['extracted_text'] = df[text_cols].bfill(axis=1).iloc[:, 0].fillna("")
        else:
            if 'confidence' not in df.columns:
                df['confidence'] = 0.0
            if 'extracted_text' not in df.columns:
                df['extracted_text'] = ""

        if 'detic_confidence' not in df.columns:
            df['detic_confidence'] = 0.0
        if 'flag' not in df.columns:
            df['flag'] = "N/A"
        if 'reason' not in df.columns:
            df['reason'] = "N/A"
        if 'identified_as' not in df.columns:
            df['identified_as'] = "Unknown"

        df['mask_id'] = df['object_id'].astype(str).str.extract(r'(mask_\d+)')[0]
        df['mask_id'] = df['mask_id'].fillna('').astype(str)
        all_masks = sorted([m for m in df['mask_id'].unique().tolist() if m])

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None

    if df is not None:
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Search & Select")

            search_query = st.text_input(
                "Search Mask ID:", placeholder="e.g., mask_0", key="search_input"
            ).strip()

            filtered_masks = (
                [m for m in all_masks if search_query.lower() in m.lower()]
                if search_query else []
            )

            st.write(f"**Selected:** {len(st.session_state.selected_masks)}")

            if search_query:
                st.write(f"**Suggestions:** {len(filtered_masks)}")
                if filtered_masks:
                    st.markdown("---")
                    for mask_id in filtered_masks:
                        is_selected = mask_id in st.session_state.selected_masks
                        if st.button(
                            f"{'✓ ' if is_selected else ''}{mask_id}",
                            key=f"btn_{mask_id}",
                            use_container_width=True,
                        ):
                            if mask_id in st.session_state.selected_masks:
                                st.session_state.selected_masks.remove(mask_id)
                            else:
                                st.session_state.selected_masks.append(mask_id)
                            st.rerun()
                else:
                    st.info("❌ No masks found")
            else:
                st.info("💡 Type mask ID to search")

            if st.session_state.selected_masks:
                st.markdown("---")
                if st.button("Clear All Selections", use_container_width=True):
                    st.session_state.selected_masks = []
                    st.rerun()

        with col_right:
            st.subheader("Annotated Image")

            display_image = image.copy()
            draw = ImageDraw.Draw(display_image)

            # Crisp, readable font sized relative to image width
            font_size = max(16, int(width / 100))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    font = ImageFont.load_default()

            masks_to_draw = (
                df[df['mask_id'].isin(st.session_state.selected_masks)]
                if st.session_state.selected_masks
                else df
            )

            for idx, row in masks_to_draw.iterrows():
                mask_id   = row['mask_id']
                x = int(row['bbox_x'])
                y = int(row['bbox_y'])
                w = int(row['bbox_width'])
                h = int(row['bbox_height'])
                is_selected = mask_id in st.session_state.selected_masks

                line_color = (139, 0, 0) if is_selected else (0, 80, 200)
                line_width = 4           if is_selected else 2

                # Bounding box
                draw.rectangle([(x, y), (x + w, y + h)], outline=line_color, width=line_width)

                # ✅ Mask ID label with colored background above the box
                draw_label(draw, x, y, str(mask_id), font, is_selected)

            # ✅ High-quality lossless display — no JPEG compression
            display_high_quality(display_image)

            if not st.session_state.selected_masks:
                st.info("👆 Type in search box to find and select masks")

            # Details panel
            if st.session_state.selected_masks:
                st.divider()
                st.subheader("Selected Masks Details")

                for mask_id in st.session_state.selected_masks:
                    mask_data = df[df['mask_id'] == mask_id]
                    if not mask_data.empty:
                        row = mask_data.iloc[0]

                        with st.expander(f"📌 {mask_id}", expanded=True):
                            col_d1, col_d2 = st.columns(2)

                            with col_d1:
                                st.write(f"**Type:** {row['identified_as']}")
                                st.write(f"**Confidence:** {float(row['confidence']):.4f}")
                                st.write(f"**Detic Conf:** {float(row['detic_confidence']):.4f}")

                            with col_d2:
                                st.write(f"**Status:** {row['flag']}")
                                st.write(f"**Reason:** {row['reason']}")

                            st.write("**Bounding Box:**")
                            st.write(
                                f"X: {int(row['bbox_x'])} | Y: {int(row['bbox_y'])} | "
                                f"W: {int(row['bbox_width'])} | H: {int(row['bbox_height'])}"
                            )

                            text = row['extracted_text']
                            if isinstance(text, str) and text.strip():
                                st.write("**Extracted Text:**")
                                st.write(f"_{text}_")

else:
    st.info("📤 Upload an image and CSV file to start")
