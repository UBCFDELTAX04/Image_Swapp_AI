import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import insightface
from insightface.app import FaceAnalysis


# Initialize FaceAnalysis model
@st.cache_resource
def load_model():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    app = FaceAnalysis(name='buffalo_l', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

app_model = load_model()

# Utility: Convert uploaded file/camera input to OpenCV image
def load_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Face alignment + warping
def align_and_warp_face(source_img, target_img, source_face, target_face):
    src_landmarks = source_face.landmark_3d_68
    tgt_landmarks = target_face.landmark_3d_68

    if src_landmarks is None or tgt_landmarks is None:
        return None

    src_pts = np.array(src_landmarks[:, :2], dtype=np.float32)
    tgt_pts = np.array(tgt_landmarks[:, :2], dtype=np.float32)

    M, _ = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, 5.0)
    warped = cv2.warpPerspective(source_img, M, (target_img.shape[1], target_img.shape[0]))
    return warped

# Blending
def blend_faces(target_img, warped_source_img, target_face_bbox):
    x1, y1, x2, y2 = map(int, target_face_bbox)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)

    mask = np.zeros(target_img.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    blended = cv2.seamlessClone(
        warped_source_img, target_img, mask, center, cv2.NORMAL_CLONE
    )
    return blended

# ------------------- Streamlit UI -------------------
st.title(" Face Swap App \n ~by Aditya Ranjan Swain ")

# User choice: Upload or Camera
st.subheader("Choose input method:")
input_method = st.radio("Select how to provide images:", ("Upload from Files", "Use Camera"))

col1, col2 = st.columns(2)

with col1:
    if input_method == "Upload from Files":
        src_file = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"])
    else:
        src_file = st.camera_input("Take Your Photo")

with col2:
    if input_method == "Upload from Files":
        tgt_file = st.file_uploader("Upload Friend's Image", type=["jpg", "jpeg", "png"])
    else:
        tgt_file = st.camera_input("Take Friend's Photo")

if src_file and tgt_file:
    src_img = load_image(src_file)
    tgt_img = load_image(tgt_file)

    st.image([cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB),
              cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)],
             caption=["Your Image", "Friend's Image"], width=300)

    if st.button("Swap Faces"):
        src_faces = app_model.get(src_img)
        tgt_faces = app_model.get(tgt_img)

        if not src_faces or not tgt_faces:
            st.error("Could not detect faces in one of the images.")
        else:
            warped = align_and_warp_face(src_img, tgt_img, src_faces[0], tgt_faces[0])
            if warped is None:
                st.error("Face alignment failed.")
            else:
                blended = blend_faces(tgt_img, warped, tgt_faces[0].bbox)
                blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

                st.image(blended_rgb, caption="Swapped Result", use_column_width=True)

                # Convert to PNG and add download button
                result_pil = Image.fromarray(blended_rgb)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="📥 Download Swapped Image",
                    data=byte_im,
                    file_name="swapped_result.png",
                    mime="image/png"
                )
