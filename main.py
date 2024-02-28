from recognition import ai_recognition

ai = ai_recognition.AiRecognition(
    camera_frame_width=1280,
    camera_frame_height=720,
    flip_code=None,
    camera_id=0,
)

if __name__ == "__main__":
    ai.camera_processing()
