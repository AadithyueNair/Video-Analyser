import cv2
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def extract_frames(video_path, output_folder):
    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0
    while success:
        # Save frame as JPEG file
        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, image)
        success, image = video_capture.read()
        count += 1
    video_capture.release()
    print(f"Extracted {count} frames")

def extract_frames_st(video_path, output_folder, progress_bar):
    # Capture the video
    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)  # Get the frame rate
    st.caption(f"Video Frame Rate: {frame_rate} FPS")  # Display the frame rate
    while success:
        # Save frame as JPEG file
        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, image)
        success, image = video_capture.read()
        count += 1
        progress = count / total_frames
        progress_bar.progress(progress)
    video_capture.release()
    st.write("Extraction Complete!")
    st.write(f"Extracted {count} frames sucessfully!")
    print(f"Extracted {count} frames")

def remove_files_from_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def resize_image(image, target_size):
    """ Resize an image to the target size. """
    return cv2.resize(image, (target_size[1], target_size[0]))

def compare_images_by_element(image1, image2):
    """ Compare two images and return True if they are the same, else False. """
    return np.array_equal(image1, image2)

def compare_images_by_norm(image1, image2):
    """
    Compare two images using norm check and return True if they are the same, else False.
    The threshold value determines how sensitive the comparison is.
    """
    threshold=0.08
    image2 = resize_image(image2, image1.shape)
    # Compute the norm of the difference between the images
    difference = cv2.norm(image1, image2, cv2.NORM_L2)
    
    # Normalize the difference by the number of elements in the image
    num_elements = image1.shape[0] * image1.shape[1]
    difference_normalized = difference / num_elements
    
    return difference_normalized < threshold, difference_normalized

def identity_checker(master_image,folder_path):
    result_dict = {}
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        # Load the image to compare
        compare_image = cv2.imread(file_path)
        # Check if the image is loaded
        if compare_image is None:
            print(f"Error loading image: {file_path}")
            continue

        status, normalised_euclidean_difference = compare_images_by_norm(master_image, compare_image)
        # Compare the master image with the current image
        if status:
            msg = 'not changed'
        else:
            msg = 'changed'
        # Print the result
        print(f"Image {filename}: {msg}, Average Euclidean Distance: {normalised_euclidean_difference}")
        #visualize_image_difference(master_image, compare_image)
        result_dict[filename] = status
    return result_dict
# Function to check identity of images in a folder against a master image
def identity_checker_st(master_image, folder_path, progress_bar):
    c=st.container(height = 200)
    result_dict = {}
    total_images = len([filename for filename in os.listdir(folder_path) if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        compare_image = cv2.imread(file_path)
        if compare_image is None:
            print(f"Error loading image: {file_path}")
            continue
        status, normalised_euclidean_difference = compare_images_by_norm(master_image, compare_image)
        # Compare the master image with the current image
        if status:
            msg = 'NO CHANGES'
        else:
            msg = 'CHANGED'
        # Print the result
        c.write(f"Image {filename}:    {msg},         Average Euclidean Distance: {round(normalised_euclidean_difference,3)}")
        print(f"Image {filename}: {msg}, Average Euclidean Distance: {normalised_euclidean_difference}", "Current Threshold: 0.08")
        result_dict[filename] = (status, normalised_euclidean_difference)
        progress = len(result_dict) / total_images
        progress_bar.progress(progress)
    return result_dict

def visualize_image_difference(image1, image2):
    """
    Visualize the difference between two images.
    """
    image2 = resize_image(image2, image1.shape)
    difference = cv2.absdiff(image1, image2)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Image 1')
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title('Image 2')
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 3)
    plt.title('Difference')
    plt.imshow(difference, cmap='gray')
    plt.show()