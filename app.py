import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from PIL import ImageFont

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
        
        # Ensure required columns exist
        required_cols = ['object_id', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col != 'object_id' else ''
        
        # Handle rotation-based CSV format
        rotation_cols = [
            'confidence_0deg',
            'confidence_90deg',
            'confidence_180deg',
            'confidence_270deg'
        ]
        
        if all(col in df.columns for col in rotation_cols):
            df['confidence'] = df[rotation_cols].max(axis=1)
            text_cols = [
                'extracted_text_0deg',
                'extracted_text_90deg',
                'extracted_text_180deg',
                'extracted_text_270deg'
            ]
            df['extracted_text'] = df[text_cols].bfill(axis=1).iloc[:, 0].fillna("")
        else:
            if 'confidence' not in df.columns:
                df['confidence'] = 0.0
            if 'extracted_text' not in df.columns:
                df['extracted_text'] = ""
        
        # Add default columns if missing
        if 'detic_confidence' not in df.columns:
            df['detic_confidence'] = 0.0
        if 'flag' not in df.columns:
            df['flag'] = "N/A"
        if 'reason' not in df.columns:
            df['reason'] = "N/A"
        if 'identified_as' not in df.columns:
            df['identified_as'] = "Unknown"
        
        # Extract mask_id (e.g., "mask_0" from "BrandingActivity_13516_09122025PM47811265_mask_0")
        df['mask_id'] = df['object_id'].str.extract(r'(mask_\d+)')[0]
        
        # Get all unique mask IDs sorted
        all_masks = sorted(df['mask_id'].unique().tolist())
            
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None
    
    if df is not None:
        # Layout
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.subheader("Search & Select")
            
            # Search box with smart suggestions
            search_query = st.text_input("Search Mask ID:", placeholder="e.g., mask_0", key="search_input").strip()
            
            # Filter masks based on search with autocomplete
            if search_query:
                filtered_masks = [m for m in all_masks if search_query.lower() in m.lower()]
            else:
                filtered_masks = []
            
            # Display statistics
            st.write(f"**Selected:** {len(st.session_state.selected_masks)}")
            
            if search_query:
                st.write(f"**Suggestions:** {len(filtered_masks)}")
                
                # Display autocomplete suggestions as clickable buttons
                if filtered_masks:
                    st.markdown("---")
                    for mask_id in filtered_masks:
                        is_selected = mask_id in st.session_state.selected_masks
                        
                        if st.button(
                            f"{'✓ ' if is_selected else ''}{mask_id}",
                            key=f"btn_{mask_id}",
                            use_container_width=True
                        ):
                            # Toggle selection
                            if mask_id in st.session_state.selected_masks:
                                st.session_state.selected_masks.remove(mask_id)
                            else:
                                st.session_state.selected_masks.append(mask_id)
                            st.rerun()
                else:
                    st.info("❌ No masks found")
            else:
                st.info("💡 Type mask ID to search")
            
            # Clear button
            if st.session_state.selected_masks:
                st.markdown("---")
                if st.button("Clear All Selections", use_container_width=True):
                    st.session_state.selected_masks = []
                    st.rerun()
        
        with col_right:
            st.subheader("Annotated Image")
            
            # Draw image with bounding boxes
            display_image = image.copy()
            draw = ImageDraw.Draw(display_image)
            
            # Use smaller, clear font
            font_size = max(14, int(width / 120))
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Determine which masks to draw
            if st.session_state.selected_masks:
                # If masks are selected, draw only selected ones
                masks_to_draw = df[df['mask_id'].isin(st.session_state.selected_masks)]
            else:
                # If no masks selected, draw all
                masks_to_draw = df
            
            # Draw bounding boxes
            for idx, row in masks_to_draw.iterrows():
                mask_id = row['mask_id']
                x = int(row['bbox_x'])
                y = int(row['bbox_y'])
                w = int(row['bbox_width'])
                h = int(row['bbox_height'])
                
                # Determine line color based on selection
                if mask_id in st.session_state.selected_masks:
                    # Dark red for selected boxes
                    line_color = (139, 0, 0)
                    line_width = 4
                else:
                    # Blue for unselected boxes
                    line_color = (0, 0, 255)
                    line_width = 2
                
                # Draw rectangle
                draw.rectangle(
                    [(x, y), (x + w, y + h)],
                    outline=line_color,
                    width=line_width
                )
                
                # Draw label with dark black text, no background
                label_text = str(mask_id)
                draw.text((x + 3, y + 3), label_text, font=font, fill=(0, 0, 0))
            
            st.image(display_image, use_container_width=True)
            
            if not st.session_state.selected_masks:
                st.info("👆 Type in search box to find and select masks")
            
            # Show details for selected masks
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
                            st.write(f"X: {int(row['bbox_x'])} | Y: {int(row['bbox_y'])} | W: {int(row['bbox_width'])} | H: {int(row['bbox_height'])}")
                            
                            text = row['extracted_text']
                            if isinstance(text, str) and text.strip():
                                st.write("**Extracted Text:**")
                                st.write(f"_{text}_")

else:
    st.info("📤 Upload an image and CSV file to start")
