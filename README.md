# Age-and-gender-detection-system-
Dataset Used: The UTKFace dataset was downloaded and then uploaded under the name
"archive.zip."

Model Architecture:
1. Input Layer: The model takes grayscale images of size 128x128 pixels as input.
2. Convolutional Layers: Utilizes Conv2D layers with increasing filter sizes (64, 128, 256)
and max-pooling to extract features from the input images.
3. Flatten Layer: Flattens the output from the convolutional layers to prepare for the
subsequent dense layers.
4. Dense Layers: Contains a dense layer with 512 units and ReLU activation function to
learn complex patterns in the flattened features.
5. Dropout Layer: Introduces dropout regularization to prevent overfitting by randomly
setting a fraction of input units to zero during training.
6. Transformer Block: Incorporates a Multi-Head Attention mechanism with 7 heads and
Layer Normalization to enhance feature interactions and learning.
7. BiLSTM Block: Employs a Bidirectional LSTM layer with 64 units and
GlobalMaxPooling1D to capture temporal dependencies in the transformed features.
8. Output Layers: Consists of two dense layers for gender and age prediction.
The total time taken by the model to run for 10 epochs is approximately 6 minutes and 9
seconds.

Model Performance Metrics:
Total Loss: 13.40191650390625
Gender Loss: 6.700958251953125
Age Loss: 6.700958251953125
Gender Accuracy: 0.6655419654154365
Age MAE: 0.620071530342102
Age MSE: 6.0808868408203125

Real-Time Face Detection and Age/Gender Prediction
How it Works
Start Streaming: The video_stream() function is called to start streaming video from the
webcam.
Frame Processing: The code continuously receives frames from the webcam using
video_frame() and converts them to OpenCV images.
Face Detection: It uses a Haar cascade classifier (face_cascade) to detect faces in the
grayscale image.
Prediction: For each detected face, it extracts the region, resizes it to (128, 128), and
reshapes it for the model. Then, it predicts the gender and age using a pre-trained model
(model.predict()).
Display: It overlays a bounding box around each detected face and displays the
predicted gender and age near the bounding box.
