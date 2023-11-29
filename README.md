# ComputerVisionProject

##Revolutionizing Cybersickness Prediction: Harnessing Transfer Learning to Predict Cybersickness Using VR HMD Stereo Images

###Abstract
The growing popularity of VR has led to its diverse applications in healthcare, education, military training, and more. However, concerns have emerged about cybersickness, hindering prolonged VR exposure. Researchers have devised various mitigation techniques, with cybersickness prediction proving effective. This typically involves collecting data like head tracking, eye tracking, stereoscopic video, and biophysiological measures. While the latter requires external sensors and restricts user movement, past studies achieved remarkable success in cybersickness prediction using head and eye tracking data readily available in modern HMDs. However, There has been limited research exploring the utilization of stereoscopic video data for predicting cybersickness. Our research introduces a novel transfer learning approach, demonstrating significant success in analyzing VR video scenes and outperforming previous approaches with a 68.44% accuracy rate.

###Introduction
In recent years, the popularity of virtual reality (VR) has surged due to advances in head-mounted displays (HMDs), boasting features like smooth 90Hz refresh rates, high-resolution stereoscopic images (2160 x 2160 per eye), and a broad 180-degree field of view. However, the immersive VR experience often triggers motion sickness, known as cybersickness or visually induced motion sickness (VIMS), causing discomforts such as dizziness, nausea, and eyestrain, hindering prolonged VR use. Real-time cybersickness prediction holds promise for automating mitigation techniques during VR exposure. Researchers have primarily used methods like Head and Eye Tracking data provided by HMDs and biophysiological measures (e.g., Heart Rate, Galvanic Skin Response) collected through additional sensors to predict cybersickness. However, collecting biophysiological data can restrict user movement. Utilizing readily available HMD data is desirable, and Head and Eye Tracking data has shown success. Nevertheless, predicting cybersickness using stereoscopic video data has been less explored, with limited research addressing the challenges associated with learning from video data in this context. In our research project, we tackle these challenges by utilizing transfer learning to predict cybersickness through the analysis of stereoscopic video data.

###Motivation & Background
Recent studies have investigated the use of stereo-image datasets from VR videos to predict cybersickness. Padmanaban et al. utilized a dataset of 19 VR videos, achieving a root-mean-square error (RMSE) of 12.6 with depth and optical flow features. Lee et al. improved this with a 3D-convolutional neural network, reaching an RMSE of 8.49, incorporating optical flow, disparity, and saliency features. Kim et al. used a convolutional autoencoder with exceptional motion videos. Prior studies often used pre-recorded HMD-rendered videos, limiting participant interaction. Jin et al. achieved an 86.8% R^2 value using diverse VR videos. In this study, we used transfer learning for extracting features from stereo images in virtual sessions with a larger, diverse ImageNet dataset, enhancing understanding through knowledge transfer from a broader image range.
![image](https://github.com/Jyotinag/ComputerVisionProject/assets/33356879/25f8d5d6-65ef-4aff-99ac-2946152d3cc0)


![image](https://github.com/Jyotinag/ComputerVisionProject/assets/33356879/b320aa33-1daa-4498-aae0-c738148a4416)

![image](https://github.com/Jyotinag/ComputerVisionProject/assets/33356879/4f0c6c0c-260d-4d04-99ab-a73405a41291)

