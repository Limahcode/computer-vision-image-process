# # import streamlit as st

# # st.title("Hello Streamlit 👋")
# # st.write("This is my first Streamlit app!")

# import streamlit as st
# import cv2
# import numpy as np
# from PIL import Image

# st.title("Computer Vision Assignment - Image Processing App")
# st.markdown("Upload an image and apply various image processing techniques.")

# # --- Upload Image ---
# uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     image_np = np.array(image.convert("RGB"))  # PIL to NumPy
#     image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     st.subheader("Original Image")
#     st.image(image, use_container_width=True)

#     # --- Technique Selection ---
#     st.subheader("🛠️ Choose Image Processing Technique")
#     technique = st.selectbox("Select a technique:", [
#         "None",
#         "Thresholding",
#         "Smoothing / Blurring",
#         "Edge Detection",
#         "Contour Detection",
#         "Template Matching",
#         "Watershed Segmentation",
#         "Color Space Transformation"
#     ])

#     processed_img = image_bgr.copy()

#     if technique == "Thresholding":
#         st.markdown("**Binary Thresholding**")
#         gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#         threshold = st.slider("Threshold value", 0, 255, 127)
#         _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
#         processed_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

#     elif technique == "Smoothing / Blurring":
#         ksize = st.slider("Kernel size", 1, 15, 5, step=2)
#         processed_img = cv2.GaussianBlur(processed_img, (ksize, ksize), 0)

#     elif technique == "Edge Detection":
#         st.markdown("**Canny Edge Detection**")
#         lower = st.slider("Lower threshold", 0, 300, 50)
#         upper = st.slider("Upper threshold", 0, 300, 150)
#         gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray, lower, upper)
#         processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

#     elif technique == "Contour Detection":
#         gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(processed_img, contours, -1, (0, 255, 0), 2)

#     elif technique == "Template Matching":
#         st.markdown("**Click on a region in the original image to use as a template.**")
#         st.info("For now, a dummy rectangle is used as a template.")
#         template = processed_img[50:150, 50:150]  # Dummy template
#         res = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#         top_left = max_loc
#         h, w = template.shape[:2]
#         cv2.rectangle(processed_img, top_left, (top_left[0] + w, top_left[1] + h), (0, 0, 255), 2)

#     elif technique == "Watershed Segmentation":
#         st.markdown("**Watershed Segmentation**")
#         gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#         # Noise removal
#         kernel = np.ones((3, 3), np.uint8)
#         opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

#         # Background and foreground
#         sure_bg = cv2.dilate(opening, kernel, iterations=3)
#         dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
#         _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

#         # Unknown region
#         sure_fg = np.uint8(sure_fg)
#         unknown = cv2.subtract(sure_bg, sure_fg)

#         # Marker labeling
#         _, markers = cv2.connectedComponents(sure_fg)
#         markers = markers + 1
#         markers[unknown == 255] = 0

#         markers = cv2.watershed(processed_img, markers)
#         processed_img[markers == -1] = [255, 0, 0]

#     elif technique == "Color Space Transformation":
#         option = st.selectbox("Choose color space:", ["HSV", "Grayscale", "LAB", "YCrCb"])
#         if option == "HSV":
#             processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
#         elif option == "Grayscale":
#             gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
#             processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#         elif option == "LAB":
#             processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
#         elif option == "YCrCb":
#             processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YCrCb)

#     # --- Display Processed Image ---
#     st.subheader("🔍 Processed Image")
#     result_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
#     st.image(result_rgb, use_container_width=True)

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Computer Vision Assignment - Image Processing App")
st.markdown("Upload an image or use the default one to apply various image processing techniques.")

# --- Upload Image or Use Default ---
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.success(" Image uploaded successfully!")
else:
    st.info(" No image uploaded. Using default sample image.")
    image = Image.open("sample.jpg")  # Make sure this file is in the same folder

# Convert to NumPy format (for OpenCV)
image_np = np.array(image.convert("RGB"))  # PIL to NumPy
image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

st.subheader("Original Image")
st.image(image, use_container_width=True)

# --- Technique Selection ---
st.subheader(" Choose Image Processing Technique")
technique = st.selectbox("Select a technique:", [
    "None",
    "Thresholding",
    "Smoothing / Blurring",
    "Edge Detection",
    "Contour Detection",
    "Template Matching",
    "Watershed Segmentation",
    "Color Space Transformation"
])

processed_img = image_bgr.copy()

if technique == "Thresholding":
    st.markdown("**Binary Thresholding**")
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    threshold = st.slider("Threshold value", 0, 255, 127)
    _, result = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    processed_img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

elif technique == "Smoothing / Blurring":
    ksize = st.slider("Kernel size", 1, 15, 5, step=2)
    processed_img = cv2.GaussianBlur(processed_img, (ksize, ksize), 0)

elif technique == "Edge Detection":
    st.markdown("**Canny Edge Detection**")
    lower = st.slider("Lower threshold", 0, 300, 50)
    upper = st.slider("Upper threshold", 0, 300, 150)
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, lower, upper)
    processed_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

elif technique == "Contour Detection":
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(processed_img, contours, -1, (0, 255, 0), 2)

elif technique == "Template Matching":
    st.markdown("**Template Matching**")
    st.info("Using a dummy template from the image (region 50:150, 50:150).")
    template = processed_img[50:150, 50:150]  # Dummy template
    res = cv2.matchTemplate(processed_img, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    h, w = template.shape[:2]
    cv2.rectangle(processed_img, top_left, (top_left[0] + w, top_left[1] + h), (0, 0, 255), 2)

elif technique == "Watershed Segmentation":
    st.markdown("**Watershed Segmentation**")
    gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(processed_img, markers)
    processed_img[markers == -1] = [255, 0, 0]

elif technique == "Color Space Transformation":
    option = st.selectbox("Choose color space:", ["HSV", "Grayscale", "LAB", "YCrCb"])
    if option == "HSV":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
    elif option == "Grayscale":
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif option == "LAB":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2LAB)
    elif option == "YCrCb":
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2YCrCb)

# --- Show Processed Image ---
st.subheader(" Processed Image")
result_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
st.image(result_rgb, use_container_width=True)

