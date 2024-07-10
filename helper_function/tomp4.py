import os
import cv2
import re

def extract_number(filename):
    # Extracts a number from a filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return -1

def images_to_video(image_folder, video_name, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png")]
    images.sort(key=lambda x: extract_number(x))

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    folder_path = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Public\Public\rain-break'  # Replace with your folder path
    video_name = r'I:\Werkstudenten\Deepak_Raj\DATASETS\Public\mp4\rain-break.mp4'
    images_to_video(folder_path, video_name)
