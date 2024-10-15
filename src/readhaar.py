import cv2

# Load the Haar Cascade file
cascade_path = './src/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Check if the cascade was loaded successfully
if face_cascade.empty():
    print(f"Error: Unable to load Haar Cascade file at {cascade_path}. Please check the file path and ensure the file exists.")
else:
    # Load the image
    image_path = 'C:/Users/MedAyoub/Desktop/face/Emotion-detection/src/data/test/angry/35669.jpg'
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Please check the file path and integrity.")
    else:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Check if any faces were detected
        if len(faces) == 0:
            print("No faces detected.")
        else:
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Display the result
            cv2.imshow('Faces', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
