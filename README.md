# AI Weather Nowcast

## Project Overview
This project develops an AI-based weather nowcasting system to predict short-term (0-2 hour) precipitation at a hyperlocal scale (1-2 km resolution) for India. Inspired by the India Meteorological Department's (IMD) needs, it adapts techniques from USA NEXRAD radar data due to limited historical IMD archives. The system uses a Convolutional Long Short-Term Memory (ConvLSTM) neural network to process sequences of radar images and predict future precipitation frames, targeting a hackathon demo on August 24-25, 2025.

## Features
- **Input**: 6 sequential radar images (128x128 pixels, 1 channel) to predict the next frame.
- **Model**: ConvLSTM2D with 64 filters (3x3 kernel) x3, BatchNormalization, and a final Conv2D layer (1 filter, sigmoid activation).
- **Output**: Predicted radar frame, with iterative predictions for up to 12 frames.
- **Metrics**: Mean Absolute Error (MAE) and Mean Squared Error (MSE) for model evaluation.
- **Goal**: Achieve hyperlocal accuracy comparable to IMD nowcasts (e.g., Uttarakhand heavy rain alert, August 16, 2025).
## Project Status (Updated: August 25, 2025, 12:41 PM IST)

### Current Progress
- **Data Preprocessing**: Successfully generated training sequences using `create_sequences.py`. The dataset now includes 245 sequences, each with 6 input frames (128x128x1) and 6 target frames (128x128x1), saved as `data/processed/sequences/train_x.npy` and `train_y.npy`.
- **Model Training**: Completed training of the ConvLSTM model using `train_model.py` on August 20, 2025, over 20 epochs. The model architecture includes ConvLSTM2D layers (150,016 params), BatchNormalization, Dropout, and a Conv3D output layer (total params: 261,985). Training ran on a CPU-only setup (gayatri-Vostro-3400) with an average epoch time of ~288-314 seconds.
- **Training Metrics**:
  - Best Epoch (20): Validation Loss 0.0110, Validation MAE 0.0451
  - Final Model: Saved as `models/nowcast_model_mumbai_v2.h5`
  - Steady improvement in MAE from 0.2020 (Epoch 1) to 0.0451 (Epoch 20), indicating robust learning.
- **Testing**: Pending execution of `test_model.py` to generate prediction outputs (e.g., `prediction.png`).
- **Artifacts**: Training history visualization and prediction comparison images are yet to be generated due to the ongoing testing phase.

### Project Goals
- Train a ConvLSTM model to predict the next 6 radar frames from 6 prior frames using iterative prediction.
- Generate and save the trained model (`nowcast_model_mumbai_v2.h5`), a training history plot (`training_history.png`), and a prediction comparison image (`prediction.png`).
- Optimize for hyperlocal accuracy, aligning with IMD nowcasts (e.g., Mumbai heavy rain alerts).
- Prepare a demo for the hackathon showcasing real-time precipitation predictions.

## Visualizations
| Image | Description |
|-------|-------------|
| <img width=400 src="https://github.com/user-attachments/assets/0780c60c-11ff-462e-8d51-35f47005823c"/> | User interface for the weather nowcasting system. |
| <img width="200"  alt="image" src="https://github.com/user-attachments/assets/98338a5f-a3ac-4aa8-8182-061c1f68a3e1"/> | Predicted radar frame for precipitation. |
| <img width="281" height="275" alt="image" src="https://github.com/user-attachments/assets/2da1b751-31a1-4751-9d79-afca45b13401" /> | Actual radar frame for comparison. |
| <img width=500 src="https://github.com/user-attachments/assets/b040689f-d733-414e-8091-0c0bf07cd07c"/> | Sequence of images and predicted as well as actual img |
| <img width=400 src="https://github.com/user-attachments/assets/e0256e22-4328-4bed-9621-25f5dad936ca"/> | Training and validation loss/MAE over epochs. |

## Repository Structure

```
AI_Weather_Nowcast/
├── backend/                    # Backend code for model inference and API
│   ├── api/                   # API endpoints for model predictions
│   │   ├── predict.py         # Handles prediction requests
│   │   ├── app.py            # Main Flask/FastAPI application
│   ├── config/                # Configuration files
│   │   ├── config.yaml       # Backend configuration (e.g., model path, API settings)
│   ├── tests/                # Backend unit tests
│   │   ├── test_api.py       # Tests for API endpoints
├── frontend/                   # React-based frontend for visualization
│   ├── src/                  # React source code
│   │   ├── components/       # Reusable React components
│   │   │   ├── RadarDisplay.js  # Component for radar image visualization
│   │   ├── App.js            # Main React application
│   │   ├── index.js          # Entry point for React app
│   ├── public/               # Static files (e.g., index.html)
│   ├── package.json          # Frontend dependencies and scripts
├── data/
│   ├── processed/
│   │   ├── preprocessed/     # Normalized radar images
│   │   ├── sequences/        # Input (.npy) and output (.npy) sequences
├── images/
│   ├── ui_image.png          # User interface screenshot
│   ├── prediction.png        # Predicted radar frame
│   ├── actual_image.png      # Actual radar frame
│   ├── training_history.png  # Training history plot
├── models/
│   ├── nowcast_model.h5      # Trained ConvLSTM model (to be regenerated)
├── scripts/
│   ├── preprocess_images.py  # Normalizes radar images
│   ├── create_sequences.py   # Generates input-output sequences
│   ├── train_model.py        # Defines and trains ConvLSTM model
│   ├── test_model.py         # Predicts 12 frames using trained model
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies

```

## Dependencies

### Backend:

TensorFlow (CPU-only)
NumPy
Flask/FastAPI (for API)
Python 3.10

### Frontend:

React
Tailwind CSS
Node.js


## Purpose:

Weather nowcasting is crucial because it provides fast and accurate updates about sudden weather changes, which can save lives and protect property. In India, where heavy rains and cloudbursts often lead to floods, landslides, and other disasters—especially in cities like Mumbai, Chennai, and hilly regions—having a reliable prediction system is essential. Traditional weather forecasts can be too slow or vague for short-term events, but this AI tool will offer real-time warnings for the next 0-2 hours. This can help:

i. Save Lives: Give people and authorities time to move to safety or prepare for floods within the critical 2-hour window.

ii. Protect Property: Help farmers, businesses, and cities minimize damage from sudden storms.

iii. Support Decision-Making: Assist the India Meteorological Department (IMD) and local governments in planning quick evacuations or emergency actions.

iv. Improve Safety: Reduce risks for people in vulnerable areas by predicting dangerous weather before it strikes.


By tailoring the system to India’s unique weather patterns and using local radar data, this project aims to make weather predictions more reliable and actionable, especially during the monsoon season when extreme weather is frequent.

## Model Architecture

Layers: 3 ConvLSTM2D layers (64 filters, 3x3 kernel), each followed by BatchNormalization, and a final Conv2D layer (1 filter, sigmoid activation).
Parameters: 741,697
Training Config: Epochs=5, batch_size=4, validation_split=0.2, optimizer='adam', loss='mse', metrics=['mae']

## Future Work

Frontend: Develop a React-based interface for visualizing predictions (post-model training).
Backend: Implement a backend for real-time data processing and model inference.
Optimization: Improve model accuracy for hyperlocal predictions, validated against IMD nowcasts.

