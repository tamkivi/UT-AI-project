# UT-AI-project

Flask web app for classifying waste images with a Keras model. Users upload a photo and the app returns the predicted class, confidence, and a short disposal tip.

## Key files
- `main.py`: Flask app, image preprocessing, and inference.
- `keras_model.h5`: Trained Keras model.
- `labels.txt`: Model class labels.
- `static/` and `templates/`: Frontend assets.
- `requirements.txt`: Python dependencies.

## Run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python main.py
   ```
3. Open http://localhost:5000

## API
- `GET /`: Upload form UI.
- `POST /upload`: Accepts `image` file upload and returns JSON with class, confidence, and base64 preview.
