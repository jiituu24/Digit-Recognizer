# Digit Detection with Neural Networks

This project implements a simple neural network for handwritten digit classification using TensorFlow and Keras. It is designed to train on digit image data and predict digit labels for new samples.

## Project Structure

- `Model.py` / `Model.ipynb`: Defines and trains the neural network model.
- `model_training.ipynb`: End-to-end workflow for training and prediction.
- `train.csv`: Training data (first column: label, remaining columns: pixel values).
- `test.csv`: Test data (pixel values only).
- `sample_submission.csv`: Example format for submission.

## How It Works

1. **Data Loading**: Reads CSV files and prepares training and test datasets.
2. **Model Definition**: Uses a simple feedforward neural network with two hidden layers.
3. **Training**: Trains the model on the provided training data.
4. **Prediction**: Predicts digit labels for test data and outputs results.

## Usage

### Requirements
- Python 3.7+
- TensorFlow
- NumPy

Install dependencies:
```bash
pip install tensorflow numpy
```

### Training the Model
Run the notebook `Model.ipynb` or execute `Model.py` (if implemented as a script) to train the model on `train.csv`.

### Making Predictions
Use `model_training.ipynb` to load the trained model and predict on `test.csv`. Results are printed and can be saved for submission.

## Example
```python
# Load and train model
model.fit(X, Y, epochs=100)

# Predict
prediction = tf.nn.softmax(model.predict(xt)).numpy()
```

## Contributing
Feel free to fork the repository and submit pull requests for improvements or bug fixes.

## License
This project is open source and available under the MIT License.
