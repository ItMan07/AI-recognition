import people.people_recognition
from face.face_recognition import AiFace

# ai = AiFace(
#     camera_frame_width=1280,
#     camera_frame_height=720,
#     flip_code=None,
#     camera_id=0,
# )

if __name__ == "__main__":
    # ai.processing()
    # ai.image_recognize("image.png")
    people.people_recognition.people_recognition()
