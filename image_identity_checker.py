import cv2
import os, shutil
import numpy as np
from tempfile import NamedTemporaryFile
import streamlit as st
import matplotlib.pyplot as plt
from utils import *

# Ensure the directory for uploaded videos exists
upload_folder = "uploaded_videos"
output_folder = "extracted_frames"
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

#Streamlit UI
st.sidebar.title("Video Analyzer")
master_image_file = st.sidebar.file_uploader("Add Master Image", type=['png', 'jpg', 'jpeg'])
video_file = st.sidebar.file_uploader("Add Video", type=['mp4', 'mov', 'avi'])

st.title("Video Analyzer")
# Create two columns
col1, col2 = st.columns(2)

# Display the uploaded image in the first column
with col1:
    if master_image_file is not None:
        st.image(master_image_file, caption='Uploaded Image', use_column_width=True)
    else:
        st.write("Please upload an image.")

# Display the uploaded video in the second column
with col2:
    if video_file is not None:
        st.video(video_file)
    else:
        st.write("Please upload a video.")

st.divider()

if st.sidebar.button("Check"):
    if master_image_file is None or video_file is None:
        st.error("Please upload both the master image and the video.")
    else:
        # Load master image
        master_image = cv2.imdecode(np.fromstring(master_image_file.read(), np.uint8), 1)

        if master_image is None:
            st.error("Error loading master image.")
        else:
            # Save the uploaded video to the upload folder
            video_path = os.path.join(upload_folder, video_file.name)
            with open(video_path, 'wb') as f:
                f.write(video_file.read())

            # Remove existing files in the output folder
            remove_files_from_folder(output_folder)

             # Extract frames from the video
            st.header("Extracting Frames from video ...")
            progress_bar_extract = st.progress(0)
            extract_frames_st(video_path, output_folder, progress_bar_extract)
            st.divider()

            # Check identity of frames against the master image
            st.header("Image Analyser checking if there are any identical images ....")
            st.subheader("Printing ...")
            st.caption("Euclidean Threshold: 0.08")
            progress_bar_check = st.progress(0)
            result_dict = identity_checker_st(master_image, output_folder, progress_bar_check)
            print(result_dict)
            distances = [dist for _,dist in result_dict.values()]
            print("Distances: {}".format(distances))
            st.divider()
            st.header("Histogram of the euclidean distances:")
            # Plotting the histogram
            plt.hist(distances, bins=30, edgecolor='black')
            plt.title('Distribution of Euclidean Distances')
            plt.xlabel('Euclidean Distance')
            plt.ylabel('Frequency')
            plt.show()
            # Displaying the histogram in Streamlit
            st.pyplot(plt)

