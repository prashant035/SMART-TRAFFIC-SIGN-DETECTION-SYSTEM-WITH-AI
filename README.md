# SMART-TRAFFIC-SIGN-DETECTION-SYSTEM-WITH-AI
"Smart Traffic Sign Detection System using Deep Learning and OpenCV. 🚦 This project detects and recognizes traffic signs in real-time, helping in road safety, autonomous driving, and intelligent transportation systems."

An AI-powered Smart Traffic Sign Detection System built with YOLOv8.
This system can detect and classify traffic signs in real-time using a webcam/camera and provides audio feedback through a speaker.
📌 Features
✅ Real-time traffic sign detection
✅ Trained on multiple datasets for robust performance
✅ Cross-platform (Mac, Windows, Linux)
✅ Voice alerts for detected traffic signs
✅ Easy to set up and run with requirements.txt
🖥️ System Requirements
CPU / GPU (GPU recommended for faster inference)
Working Camera (Webcam / USB camera)
Working Speaker (for audio alerts)
Python 3.8+
⚙️ Installation & Setup
Clone the Repository
git clone https://github.com/prashant035/smart_traffic_sign_detection_AI.git
cd smart_traffic_sign_detection_AI
Install Requirements
pip install -r requirements.txt
Run the Detection Script
python detect.py
📂 Project Structure
smart_traffic_sign_detection_AI/
│── detect.py              # Main detection script
│── classtest.py           # Testing script
│── requirements.txt       # Python dependencies
│── data.yaml              # Dataset configuration
│── README.md              # Project documentation
│── /runs                  # Training results (optional, not uploaded fully)
│── /Dataset               # Dataset (not uploaded here, see Drive link)
📦 Dataset & Model Weights
Since GitHub has file size limitations, the dataset and trained YOLOv8 model weights (best.pt, etc.) are provided separately.
🔗 Download from Google Drive: [Insert Your Drive Link Here]
After downloading, place the dataset and weights in the appropriate project folders before running the code.
📊 Training Details
Model: YOLOv8
Framework: Ultralytics YOLOv8
Training: Done on custom datasets of traffic signs
Output: Model weights (best.pt, last.pt)
🎯 How It Works
Capture video from camera
Run YOLOv8 model on each frame
Detect and classify traffic signs
Generate real-time voice alerts for drivers
📄 Report
A detailed project report is included in the repository (PROJECT_REPORT.pdf).
🤝 Contribution
Pull requests are welcome!
For major changes, please open an issue first to discuss what you would like to change.
📧 Contact
Author: Prashant
GitHub: @prashant035
