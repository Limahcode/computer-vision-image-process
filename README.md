# 🧠 Computer Vision Image Processing App

This is a simple web app built with **Streamlit** and **OpenCV** that lets you try different image processing techniques on any uploaded image.

You can use it to:
- Apply **Thresholding** to convert an image to black and white
- Use **Blurring** to reduce noise in an image
- Detect **Edges** with the Canny algorithm
- Find and draw **Contours**
- Try **Template Matching**
- Perform **Watershed Segmentation** to separate overlapping objects
- Change the image's **Color Space** (like Grayscale, HSV, LAB, etc.)

---

## 🚀 How to Use

1. Go to the live app: [Open App](https://computer-vision-image-process-jp9gp7dufkdsxna4pbn9yp.streamlit.app/)
2. Upload an image (JPG or PNG)
3. Select a technique from the dropdown
4. Play around with the sliders and see how it changes the image in real time

---

## 🛠️ Tech Stack

- **Streamlit** – for the interactive web app
- **OpenCV (cv2)** – for image processing
- **Pillow (PIL)** – for handling image formats
- **NumPy** – for array operations

---

## 📁 Files in This Repo

- `app.py` – The main Streamlit app file
- `requirements.txt` – Python libraries needed
- `packages.txt` – System packages needed (for deployment)
- `samples/` – (Optional) folder for sample images

---

## 🙋‍♂️ Why I Built This

This project was part of a computer vision assignment. I wanted to explore different techniques and make it easy to demonstrate them visually through a simple app.

---

## 📸 Example Techniques

| Technique           | Preview |
|---------------------|---------|
| Thresholding        | Turns parts of image white or black |
| Blurring            | Smooths out the image to remove noise |
| Edge Detection      | Shows edges (outlines) in white |
| Contour Detection   | Draws green lines around shapes |
| Watershed Segmentation | Splits overlapping objects |

---

## 📬 Contact

If you have any questions or feedback, feel free to [open an issue](https://github.com/Limahcode/computer-vision-image-process/issues) or reach out on GitHub.

