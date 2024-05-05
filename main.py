import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ====================================
# Load the video
def load_video(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    return cap

# ====================================
# adjust the brightness of the image
def adjust_brightness(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    adjusted = cv.LUT(image, table)
    return adjusted

# calculate the average brightness of the frames
def calculate_average_brightness(frames):
    total_brightness = 0
    for frame in frames:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        total_brightness += np.mean(gray)
    return total_brightness / len(frames)

# adjust the brightness of the frame to the average brightness
def adjust_brightness_to_average(frame, average_brightness):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    current_brightness = np.mean(gray)
    gamma = average_brightness / current_brightness
    return adjust_brightness(frame, gamma)

# ====================================
# get frames from the video
def get_frames(cap):
    frames = []
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    # If the video is less than 500 frames, collect 26 frames
    if total_frames <= 500:
        collected_frames=26
        frame_step=(total_frames//collected_frames)+1
    else: # the interval of the collected frames is 20
        frame_step = 20

    for i in range(0, total_frames, frame_step):
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        # Apply Gaussian blur to the frame
        frame = cv.GaussianBlur(frame, (5, 5), 0)
        frames.append(frame)

    return frames

# ====================================
# Filter frames by features using SIFT
def filter_frames_SIFT(frames, min_features=50):
    sift = cv.SIFT_create()
    filtered_frames = []
    for frame in frames:
        # Detect features
        kp, des = sift.detectAndCompute(frame, None)
        # If not enough features, skip this frame
        if des is None or len(des) < min_features:
            continue
        filtered_frames.append(frame)
    return filtered_frames

# ====================================
# Crop the panorama
def crop_panorama(panorama):
    # Convert the panorama to grayscale
    stitched = cv.copyMakeBorder(panorama, 10, 10, 10, 10, cv.BORDER_CONSTANT, (0, 0, 0))
    gray = cv.cvtColor(stitched, cv.COLOR_BGR2GRAY)
    # threshold the image
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)[1]
    # find the contours
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    cv.drawContours(thresh, cnts, -1, (0, 255, 0), 3)

    # Create a mask to cover the largest contour
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv.boundingRect(cnts[0])  # 取出list中的轮廓二值图，类型为numpy.ndarray
    cv.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

    # erode the mask until the minRect is all 0
    minRect = mask.copy()
    sub = mask.copy()
    # dilate the mask before eroding
    thresh = cv.dilate(thresh, None, iterations=10)
    while cv.countNonZero(sub) > 0:
        minRect = cv.erode(minRect, None)
        sub = cv.subtract(minRect, thresh)

    # extract the minRect contour and crop
    cnts = cv.findContours(minRect, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    (x, y, w, h) = cv.boundingRect(cnts[0])
    stitched = stitched[y:y + h, x:x + w]
    # plt.subplot(2, 3, 1)
    # plt.imshow(panorama)
    # plt.title("origin panorama")
    # plt.subplot(2, 3, 2)
    # plt.imshow(thresh)
    # plt.title("thresh")
    # plt.subplot(2, 3, 3)
    # plt.imshow(mask)
    # plt.title("mask")
    # plt.subplot(2, 3, 4)
    # plt.imshow(minRect)
    # plt.title("minRect")
    # plt.subplot(2, 3, 5)
    # plt.imshow(stitched)
    # plt.title("stitched")
    # plt.show()
    return stitched

# ====================================
# stitch the panorama (basic method)
def stitch_frames(frames):
    stitcher = cv.Stitcher_create()
    # Stitch all frames
    status, panorama = stitcher.stitch(frames)
    if status != cv.Stitcher_OK:
        print("Error: Could not stitch frames.")
        exit()
    return panorama

# stitch the panorama (append method)
def stitch_frames_append(frames):
    # Split the frames into two parts
    separate_index = len(frames) // 3
    left_frames = frames[0:len(frames)-separate_index]
    right_frames = frames[separate_index:len(frames)]
    print(len(left_frames), len(right_frames))
    # Stitch the frames in each part
    panorama_left = stitch_frames(left_frames)
    panorama_right = stitch_frames(right_frames)
    # plt.subplot(1, 2, 1)
    # plt.imshow(panorama_left)
    # plt.title("panorama_left")
    # plt.subplot(1, 2, 2)
    # plt.imshow(panorama_right)
    # plt.title("panorama_right")
    # plt.show()
    return panorama_left, panorama_right

# pre_process the image before append
def pre_process_image(image):
    # Create a mask that covers the center of the image
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv.circle(mask, (image.shape[1]//2, image.shape[0]//2), min(image.shape[:2])//3, 255, -1)
    # Decrease the pixel values in the center of the image
    image = cv.addWeighted(image, 1, cv.cvtColor(mask, cv.COLOR_GRAY2BGR), -0.8, 0)
    return image

# append two images
def append_image(img1, img2):
    # ------------------------------------
    temp_img1= pre_process_image(img1)
    temp_img2= pre_process_image(img2)
    # ------------------------------------
    # Initialize the SIFT detector
    sift = cv.SIFT_create(edgeThreshold=10)
    # Detect the keypoints and compute the descriptors
    kp1, des1 = sift.detectAndCompute(temp_img1, None)
    kp2, des2 = sift.detectAndCompute(temp_img2, None)

    # brute-force matcher
    bf = cv.BFMatcher()
    # Match the descriptors
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Calculate the average x coordinate of keypoints in both images
    avg_x1 = np.mean([kp1[m.queryIdx].pt[0] for m in good_matches])
    avg_x2 = np.mean([kp2[m.trainIdx].pt[0] for m in good_matches])
    # ------------------------------------
    # Determine which image is on the left and which is on the right
    if avg_x1 < avg_x2:
        # print("img1 is on the right")
        right_img, left_img = img1, img2
        # Find the perspective transformation matrix
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    else:
        # print("img2 is on the right")
        right_img, left_img = img2, img1
        # Find the perspective transformation matrix
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    # Warp the right image to align with the left image
    img2_aligned = cv.warpPerspective(right_img, M, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
    # ------------------------------------
    img2_aligned[0:left_img.shape[0], 0:left_img.shape[1]] = left_img
    return_paranoma = img2_aligned
    # ------------------------------------
    return return_paranoma


if __name__ == '__main__':
    test_num = 3 # select from [1 ,2, 3]
    # ------------------------------------
    # 1. Load the video
    video_path = 'test_videos/test{}.mp4'.format(test_num)
    cap = load_video(video_path)
    # ------------------------------------
    # 2. Get the frames
    frames = get_frames(cap)
    # ------------------------------------
    # 3. Calculate the average brightness
    average_brightness = calculate_average_brightness(frames)
    frames = [adjust_brightness_to_average(frame, average_brightness) for frame in frames]
    # ------------------------------------
    # 4. use SIFT to filter frames
    frames = filter_frames_SIFT(frames)
    print("{} frames are collected".format(len(frames)))
    # ------------------------------------
    # 5. Stitch the frames and Crop the panorama
    if len(frames) < 27:
        panorama = stitch_frames(frames)
        final = crop_panorama(panorama)
    else:
        panorama1, panorama2 = stitch_frames_append(frames)
        panorama1 = crop_panorama(panorama1)
        panorama2 = crop_panorama(panorama2)
        panorama = append_image(panorama1, panorama2)
        final = crop_panorama(panorama)
    # Save the panorama
    cv.imwrite('results/panorama_{}.jpg'.format(test_num), final)
    cap.release()