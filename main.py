from fastapi import FastAPI, Form, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import easyocr
import io
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime
from moviepy.editor import VideoFileClip
import moviepy.video.fx.all as vfx

# Load the YOLO model
model = YOLO("models/best (1).pt")

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize FastAPI
app = FastAPI()

# CORS configuration
origins = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",  # Adjust with your frontend URL
    # Add more allowed origins as needed
]

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Function to draw bounding boxes and perform OCR
def draw_bounding_boxes_and_ocr(frame, coordinates):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
    
    for coord in coordinates:
        x, y, w, h = map(int, coord)
        top_left = (x - w // 2, y - h // 2)
        bottom_right = (x + w // 2, y + h // 2)
        color = (0, 255, 0)  # Green color for the bounding box
        thickness = 2  # Thickness of the bounding box
        frame = cv2.rectangle(frame, top_left, bottom_right, color, thickness)
        
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Perform OCR on the ROI
        if roi.size > 0:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # Convert to RGB for EasyOCR
            result = reader.readtext(roi_rgb)  # Perform OCR
            for (bbox, text, prob) in result:
                cv2.putText(frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert back to RGB for MoviePy

# Route to render the login page
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Route to handle login
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username == "a" and password == "1":
        return JSONResponse(content={"success": True})
    return JSONResponse(content={"success": False})

# Route to render the main page after login
@app.get("/index.html", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to render the video page
@app.get("/video.html", response_class=HTMLResponse)
async def video_page(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})

# Route to handle image uploads
@app.post("/upload_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        results = model.predict(frame)
        coordinates = [result.boxes.xywh.tolist() for result in results]
        frame_with_boxes_and_ocr = draw_bounding_boxes_and_ocr(frame, coordinates[0]) if coordinates else frame
        _, img_encoded = cv2.imencode('.jpg', frame_with_boxes_and_ocr)
        return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Route to handle video uploads
# Function to process video
def process_video(input_path, output_path):
    try:
        # Open the video file
        clip = VideoFileClip(input_path)

        print(f"Output video will be saved to {output_path}")

        frame_count = 0

        # Define a function to process each frame
        def process_frame(get_frame, t):
            frame = get_frame(t)
            results = model.predict(frame)
            coordinates = []
            for result in results:
                coordinates.append(result.boxes.xywh.tolist())
            
            if coordinates:  
                frame_with_boxes_and_ocr = draw_bounding_boxes_and_ocr(frame, coordinates[0])
            else:
                frame_with_boxes_and_ocr = frame
            nonlocal frame_count
            frame_count += 1
            print(f"Processed frame {frame_count}")
            return frame_with_boxes_and_ocr

        # Apply the processing function to each frame of the video
        processed_clip = clip.fl(process_frame, apply_to=['mask'])

        # Write the processed video to the output file
        processed_clip.write_videofile(output_path, codec='libx264', fps=clip.fps)
        
        print(f"Video processing complete. Video saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during video processing: {str(e)}")
        raise HTTPException(status_code=500, detail="Video is too lengthy. Please input a video of 10 seconds or less.")

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Read the uploaded file content
        video_content = await file.read()
        
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(video_content)
            temp_video_path = temp_video_file.name
        
        # Check the duration of the uploaded video
        clip = VideoFileClip(temp_video_path)
        if clip.duration > 10:
            os.remove(temp_video_path)
            raise HTTPException(status_code=400, detail="Video is too lengthy. Please input a video of 10 seconds or less.")

        # Generate a unique filename for the processed video
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        processed_video_path = f'temp/processed_{timestamp}.mp4'
        
        # Process the video
        process_video(temp_video_path, processed_video_path)

        # Remove the uploaded temporary video file
        os.remove(temp_video_path)

        print(f"Processed video saved to {processed_video_path}")

        # Return the processed video as a response
        return StreamingResponse(open(processed_video_path, "rb"), media_type="video/mp4")

    except HTTPException as e:
        raise e

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the video")

# Route to handle camera stream and detection
@app.get("/detect_camera/")
async def detect_camera():
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise HTTPException(status_code=500, detail="Unable to open camera")
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            results = model.predict(frame)
            coordinates = [result.boxes.xywh.tolist() for result in results]
            frame_with_boxes_and_ocr = draw_bounding_boxes_and_ocr(frame, coordinates[0]) if coordinates else frame
            _, img_encoded = cv2.imencode('.jpg', frame_with_boxes_and_ocr)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tobytes() + b'\r\n')
        cap.release()

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
