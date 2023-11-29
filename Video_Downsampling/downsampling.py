def downsample_video(input_video_path, output_video_path, pooling_method='max', frames_per_second=1):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frames_per_second, (frame_width, frame_height))

    frame_buffer = []  # A buffer to hold frames for pooling
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_buffer.append(frame)
        frame_count += 1

        # When the buffer is filled, apply pooling and write the result
        if len(frame_buffer) == frame_rate:
            if pooling_method == 'max':
                pooled_frame = np.max(np.stack(frame_buffer), axis=0)
            elif pooling_method == 'average':
                pooled_frame = np.mean(np.stack(frame_buffer), axis=0).astype(np.uint8)
            else:
                raise ValueError("Invalid pooling method. Use 'max' or 'average'.")

            out.write(pooled_frame)
            frame_buffer = []  # Clear the buffer

    # Write any remaining frames in the buffer
    for frame in frame_buffer:
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def video_processing_pipeline(input_video_path,final_out):
    output_video_path = 'output_video.mp4'
    
    downsample_video(input_video_path, output_video_path, pooling_method='max', frames_per_second=1)
    downsampled_video_path = 'output_video.mp4'
    cap = cv2.VideoCapture(downsampled_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the output video properties
    output_video_path = final_out
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

    # Define the number of frames to keep
    frames_to_keep = 900

    frame_count = 0

    while frame_count < frames_to_keep:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)
        frame_count += 1

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()