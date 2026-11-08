
# Malaria Cell Image Classification Using CNN

A deep learning project that classifies microscopic cell images into **Parasitized** and **Uninfected** categories using a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The project includes image preprocessing, dataset visualization, CNN model training, performance evaluation, model saving, and a Flask-based image upload and classification interface.

> **Disclaimer:** This is an educational and experimental project. The model is not medically validated and must not be used for clinical diagnosis or treatment decisions.

## Project Overview

The `malaria.ipynb` notebook implements an image classification workflow using a CNN.

The workflow includes:

1. Importing Python libraries required for image processing, data manipulation, visualization, and deep learning.
2. Loading malaria cell images from separate training and testing directories.
3. Assigning labels based on image folder names.
4. Resizing images to 150 × 150 pixels and converting them to RGB format.
5. Shuffling the training dataset.
6. Exploring class distribution using bar charts and pie charts.
7. Normalizing pixel values from 0–255 to 0–1.
8. Visualizing individual images and sample image grids.
9. Building and training a CNN model using TensorFlow/Keras.
10. Evaluating the trained model using test data.
11. Generating predictions and evaluating model performance.
12. Saving the model architecture and trained weights.
13. Integrating the model with a Flask-based web application for image classification.

## Key Features

- **Image Classification:** Classifies malaria cell images into Parasitized and Uninfected categories.
- **Image Preprocessing:** Resizes images to 150 × 150 pixels, converts images to RGB, and normalizes pixel values.
- **Dataset Visualization:** Displays random images and visualizes class distribution.
- **CNN Model Development:** Builds a Sequential CNN using convolutional, pooling, flattening, and dense layers.
- **Model Training:** Trains the model using the Adam optimizer.
- **Model Evaluation:** Calculates accuracy, precision, recall, F1-score, and confusion matrix.
- **Visualization of Results:** Uses Matplotlib and Seaborn to visualize evaluation results.
- **Model Saving:** Saves the model architecture as JSON and trained weights as an H5 file.
- **Flask Integration:** Includes Flask routes intended to support image uploading and classification through a web interface.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and training |
| OpenCV | Image loading, RGB conversion, and resizing |
| NumPy | Array manipulation and image normalization |
| Pandas | Dataset statistics and classification report formatting |
| Matplotlib | Image and chart visualization |
| Seaborn | Confusion matrix and heatmap visualization |
| Scikit-learn | Data shuffling and model evaluation metrics |
| tqdm | Progress bars during image loading |
| Flask | Web application integration |
| Jupyter Notebook | Interactive development and experimentation |

## CNN Model Architecture

The notebook uses a Sequential Convolutional Neural Network architecture.

| Layer | Configuration |
|---|---|
| Input | 150 × 150 × 3 RGB image |
| Conv2D | 32 filters, 3 × 3 kernel, ReLU activation |
| MaxPooling2D | 2 × 2 pool size |
| Conv2D | 32 filters, 3 × 3 kernel, ReLU activation |
| MaxPooling2D | 2 × 2 pool size |
| Flatten | Converts feature maps into a 1D vector |
| Dense | 128 neurons, ReLU activation |
| Output | Softmax activation |

### Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 150 × 150 |
| Batch Size | 128 |
| Number of Epochs | 10 |
| Validation Split | 20% |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Evaluation Metric | Accuracy |

> **Model Configuration Note:** The notebook currently defines an output layer with `Dense(3, activation='softmax')`, while the dataset contains only two classes. The output layer should be configured with two units (`Dense(2, activation='softmax')`) for the two-class classification task. Ensure this correction is made before training and evaluation.

## Dataset

The dataset contains microscopic cell images categorized into two classes:

- **Parasitized:** Images containing parasitized cells.
- **Uninfected:** Images containing uninfected cells.

### Dataset Folder Structure

The notebook expects the dataset to be arranged as follows:

```text
dataset/
├── train/
│   ├── Parasitized/
│   │   └── images...
│   └── Uninfected/
│       └── images...
└── test/
    ├── Parasitized/
    │   └── images...
    └── Uninfected/
        └── images...
```

The folder names must match the class names defined in the notebook:

```python
class_names = ['Parasitized', 'Uninfected']
```

The dataset is not included in this repository unless it is separately added.

### Image Preprocessing

The notebook performs the following preprocessing operations:

- Loads images using OpenCV.
- Converts images from BGR to RGB.
- Resizes images to 150 × 150 pixels.
- Converts images into NumPy arrays.
- Shuffles training images and labels.
- Normalizes pixel values by dividing by 255.0.

## Model Evaluation

The trained model is evaluated using the test dataset.

The notebook calculates the following metrics:

- Test Loss
- Test Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

The evaluation results are visualized using:

- Confusion matrix heatmap
- Classification report heatmap
- Individual image predictions
- Sample image visualizations

Actual metric values depend on the dataset and the training run. No fixed accuracy value is claimed in this README.

## Model Output Files

After executing the notebook, the following files are generated:

| File | Description |
|---|---|
| `model_vgg.json` | Stores the model architecture in JSON format |
| `model_vgg.weights.h5` | Stores the trained model weights |

These files can be used to reconstruct the trained model for prediction and web application integration.

## Flask Web Application

The notebook contains Flask code intended to provide a browser-based image classification interface.

The application includes the following routes:

| Route | Method | Description |
|---|---|---|
| `/` | GET | Displays the home page |
| `/classify` | GET | Displays the image upload page |
| `/classify` | POST | Uploads an image and performs classification |
| `/classify/<filename>` | GET | Serves the uploaded image |

### Flask Application Workflow

1. The user uploads a microscopic cell image through the web interface.
2. The image is saved in the uploads directory.
3. The image is resized to 150 × 150 pixels.
4. The image is normalized.
5. The CNN model predicts the class.
6. The predicted class and probability are returned to the results page.

### Flask Integration Requirements

The notebook's Flask code requires additional setup before it can run independently:

- Model architecture must be loaded from `model_vgg.json`.
- Model weights must be loaded from `model_vgg.weights.h5`.
- The model-loading statements in the notebook are currently commented out.
- The `model` variable must be initialized before calling `classify()`.
- The `templates/home.html` and `templates/classify.html` files must be created.
- The `uploads/` directory must exist before uploaded files are saved.
- The Flask code should be extracted into a standalone `app.py` file.

The Flask section references templates and model-loading functionality that are not fully implemented in the notebook.

## Prerequisites

Before running the project, install:

- Python
- Jupyter Notebook or JupyterLab
- pip
- The malaria cell image dataset
- A compatible TensorFlow/Keras installation
- A web browser for the Flask interface

### Install Dependencies

```bash
pip install numpy opencv-python scikit-learn seaborn matplotlib pandas tqdm tensorflow jupyter flask
```

Keras is generally included with TensorFlow. If standalone Keras imports fail, ensure the installed TensorFlow and Keras versions are compatible.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repository>.git
```

Navigate to the project directory:

```bash
cd <your-repository>
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install numpy opencv-python scikit-learn seaborn matplotlib pandas tqdm tensorflow jupyter flask
```

### 4. Configure the Dataset

Place the dataset in the project directory using the folder structure described in the Dataset section.

The notebook currently contains hard-coded Windows paths inside the `load_data()` function.

Update the paths according to your local dataset location.

Example:

```python
datasets = [
    r'path/to/dataset/train',
    r'path/to/dataset/test'
]
```

Also update the hard-coded path in the single-image prediction cell.

### 5. Run the Jupyter Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `malaria.ipynb` and execute the cells in sequence:

1. Import packages
2. Load the dataset
3. Preprocess images
4. Visualize the dataset
5. Build the CNN model
6. Train the model
7. Evaluate the model
8. Generate predictions
9. Save the model architecture and weights

### 6. Run the Flask Application

After extracting the Flask code into `app.py`, ensure the following files and directories are available:

```text
model_vgg.json
model_vgg.weights.h5
app.py
templates/
├── home.html
└── classify.html
uploads/
```

Enable and implement model reconstruction and weight loading in `app.py` before starting the application.

Run the Flask application:

```bash
python app.py
```

Open the local URL displayed in the terminal.

> The notebook currently uses Flask's development server through `app.run()`. It is intended for local experimentation, not production deployment.

## Project Structure

```text
Malaria-Cell-Image-Classification/
│
├── malaria.ipynb
├── README.md
├── model_vgg.json              # Generated after model training
├── model_vgg.weights.h5        # Generated after model training
├── app.py                      # Flask application (to be extracted)
│
├── templates/
│   ├── home.html               # Required Flask template
│   └── classify.html           # Required Flask template
│
├── uploads/                    # Uploaded images
│
└── dataset/
    ├── train/
    │   ├── Parasitized/
    │   └── Uninfected/
    │
    └── test/
        ├── Parasitized/
        └── Uninfected/
```

> Some files and folders shown above are generated or referenced by the notebook and may not currently exist in the repository.

## What I Learned

- Understanding image classification using deep learning.
- Loading and preprocessing image datasets using OpenCV.
- Working with NumPy arrays and image normalization.
- Building CNN architectures using TensorFlow/Keras.
- Understanding convolutional layers, pooling layers, flattening, and dense layers.
- Training and validating a deep learning model.
- Visualizing image datasets and class distributions.
- Evaluating classification performance using precision, recall, F1-score, accuracy, and confusion matrices.
- Performing predictions on individual images and test datasets.
- Saving model architecture and trained weights.
- Understanding the integration of a CNN model with a Flask web application.

## Future Improvements

- Correct the output layer to match the two target classes.
- Replace hard-coded Windows paths with configurable relative paths.
- Implement complete model reconstruction and weight loading in Flask.
- Add the required HTML templates for the web interface.
- Add validation for uploaded image files.
- Handle missing files and invalid image formats.
- Implement a more user-friendly classification results page.
- Apply data augmentation to improve generalization.
- Add early stopping and model checkpointing.
- Explore transfer learning using pretrained CNN architectures.
- Add a `requirements.txt` file for dependency management.
- Improve reproducibility by documenting dataset acquisition and setup.

## Disclaimer

This project is intended for educational purposes and experimentation with computer vision and deep learning.

The model has not been medically validated. Its predictions should not be used to diagnose malaria, make clinical decisions, or guide medical treatment.
