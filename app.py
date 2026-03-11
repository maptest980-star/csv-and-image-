import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


st.set_page_config(layout="wide")
st.title("Image Annotation Viewer")

# Upload section
col1, col2 = st.columns(2)
with col1:
    image_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
with col2:
    annotation_file = st.file_uploader("Upload CSV", type=["csv"])

if image_file and annotation_file:
    # Load image and CSV
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image)  # fixes orientation using EXIF
    image = image.convert("RGB")
    img_array = np.array(image)
    height, width, _ = img_array.shape
    
    try:
        df = pd.read_csv(annotation_file)
        
        # Check if this is rotation-based CSV or single-column CSV
        rotation_cols = [
            'confidence_0deg',
            'confidence_90deg',
            'confidence_180deg',
            'confidence_270deg'
        ]
        
        if all(col in df.columns for col in rotation_cols):
            # Type 1: Rotation-based CSV format
            df['confidence'] = df[rotation_cols].max(axis=1)
            
            text_cols = [
                'extracted_text_0deg',
                'extracted_text_90deg',
                'extracted_text_180deg',
                'extracted_text_270deg'
            ]
            # Get the first non-empty text across rotations, or empty string if all are empty
            df['extracted_text'] = df[text_cols].bfill(axis=1).iloc[:, 0].fillna("")
        else:
            # Type 2: Single-column CSV format
            # Ensure confidence column exists
            if 'confidence' not in df.columns:
                df['confidence'] = 0.0
            
            # Ensure extracted_text column exists
            if 'extracted_text' not in df.columns:
                df['extracted_text'] = ""
        
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df = None
    
    if df is not None and 'object_id' in df.columns:
        # Layout: Left side info, Right side image
        col_left, col_right = st.columns([2, 3])
        
        # Get mask options
        mask_ids = sorted(df['object_id'].unique().tolist())
        mask_options = ["All Masks"] + mask_ids
        
        with col_left:
            st.subheader("Controls & Data")
            
            # Mask selector
            selected_mask = st.selectbox("Select Mask:", mask_options)
            
            st.divider()
            
            # Display info based on selection
            if selected_mask == "All Masks":
                st.write(f"**Total Masks:** {len(mask_ids)}")
                st.write(f"**Objects Detected:** {len(df)}")
                
                # Show summary
                if 'identified_as' in df.columns:
                    st.write("**Object Types:**")
                    obj_counts = df['identified_as'].value_counts()
                    for obj, count in obj_counts.items():
                        st.write(f"  • {obj}: {count}")
                
                st.divider()
                st.write("**All Masks List:**")
                st.dataframe(df[['object_id', 'identified_as', 'confidence']], use_container_width=True, height=300)
                
            else:
                # Single mask selection
                mask_data = df[df['object_id'] == selected_mask]
                
                if not mask_data.empty:
                    row = mask_data.iloc[0]
                    
                    # Key info
                    st.write(f"**ID:** {row['object_id']}")
                    st.write(f"**Type:** {row['identified_as']}")
                    
                    # Metrics
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Confidence", f"{float(row['confidence']):.4f}")
                    with col_m2:
                        st.metric("Detic Conf", f"{float(row['detic_confidence']):.4f}")
                    
                    st.divider()
                    
                    # Bbox info
                    st.write("**Bounding Box:**")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        st.write(f"  X: {int(row['bbox_x'])}")
                        st.write(f"  Y: {int(row['bbox_y'])}")
                    with col_b2:
                        st.write(f"  W: {int(row['bbox_width'])}")
                        st.write(f"  H: {int(row['bbox_height'])}")
                    
                    st.divider()
                    
                    # Text info
                    text = row['extracted_text']

                    if isinstance(text, str) and text.strip() != "":
                        st.write("**Extracted Text:**")
                        st.code(text)
                    else:
                        st.write("**Text:** No text detected")
                    
                    st.divider()
                    
                    # Flag info
                    st.write(f"**Status:** {row['flag']}")
                    st.caption(str(row['reason']))
        
        with col_right:
            st.subheader("Annotated Image")
            
            # Draw annotations
            display_image = image.copy()
            draw = ImageDraw.Draw(display_image)
            
            if selected_mask == "All Masks":
                # Draw all masks with dark color
                for idx, row in df.iterrows():
                    x = int(row['bbox_x'])
                    y = int(row['bbox_y'])
                    w = int(row['bbox_width'])
                    h = int(row['bbox_height'])
                    
                    # Very dark color (almost black with slight color tint)
                    draw.rectangle(
                        [(x, y), (x + w, y + h)],
                        outline=(20, 20, 50),
                        width=5
                    )
            else:
                # Draw single mask
                mask_data = df[df['object_id'] == selected_mask]
                if not mask_data.empty:
                    row = mask_data.iloc[0]
                    x = int(row['bbox_x'])
                    y = int(row['bbox_y'])
                    w = int(row['bbox_width'])
                    h = int(row['bbox_height'])
                    
                    # Very dark color
                    draw.rectangle(
                        [(x, y), (x + w, y + h)],
                        outline=(10, 10, 40),
                        width=6
                    )
                    
                    # Label
                    draw.text((x, max(0, y-15)), selected_mask, fill=(10, 10, 40))
            
            st.image(display_image, use_container_width=True)
else:

    st.info("📤 Upload an image and CSV file to start")
