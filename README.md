# Digit Recognizer with Neural Networks

This project implements a simple neural network for handwritten digit classification using TensorFlow and Keras. It is designed to train on digit image data and predict digit labels for new samples.

## Project Structure

- `Model.py` / `Model.ipynb`: Defines and trains the neural network model.
- `checking.ipynb`: Loads data, trains the model, makes predictions, and analyzes errors and cost.
- `train.csv`: Training data (first column: label, remaining columns: pixel values).
- `test.csv`: Test data (pixel values only).
- `sample_submission.csv`: Example format for submission.

## How It Works

1. **Data Loading**: Reads CSV files and prepares training and test datasets.
2. **Model Training**: Trains the model on the provided training data using `model.fit(X, Y, epochs=500)`.
3. **Prediction**: Uses `model.predict(X)` to get predictions.
4. **Error Analysis**: Compares predicted labels to true labels, prints misclassified samples.
5. **Cost Calculation**: Computes cross-entropy cost using softmax probabilities.

## Usage

### Requirements
- Python 3.7+
- TensorFlow
- NumPy

Install dependencies:
```bash
pip install tensorflow numpy
```

### Training and Evaluation
Run `checking.ipynb` to:
- Load and preprocess data
- Train the model
- Predict and analyze errors
- Compute cost

Example code:
```python
from Model import model
import tensorflow as tf
import numpy as np

# Load data
# ...see checking.ipynb for details...

# Train model
model.fit(X, Y, epochs=500)

# Predict
predictions = model.predict(X)

# Error analysis and cost calculation
# ...see checking.ipynb for details...
```

## Contributing
Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License
This project is open source and available under the MIT License.