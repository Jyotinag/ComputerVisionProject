def feature_extract(input_video):
    cap = cv2.VideoCapture('final_output_video.mp4')

    # Define new frame size
    new_width, new_height = 224, 224

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        frames.append(resized_frame)

    cap.release()
    video_data = np.array(frames)
    i3d_model = InceptionV3(weights='imagenet', include_top=False)
    # Load and preprocess your video frames
    video_data_preprocessed = preprocess_input(video_data)
    # Extract features
    features = i3d_model.predict(video_data_preprocessed)
    return features