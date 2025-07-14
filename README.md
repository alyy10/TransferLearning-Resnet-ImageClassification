# Using Transfer Learning For Image Classification

## Overview


<img width="1858" height="820" alt="Project_Architecture" src="https://github.com/user-attachments/assets/39140641-a8a8-499d-8b3c-ce80bca8041d" />



This project demonstrates the process of building an image classification model in **PyTorch** using **Transfer Learning** and a **pre-trained ResNet50 model**. The aim is to classify images into different categories such as **Social Security Cards**, **Driving Licenses**, and **Others**.

**Transfer Learning** is a technique where we reuse a pre-trained model as a starting point for a new task, leveraging the knowledge the model gained from a large, general dataset. In this case, **ResNet50**, pre-trained on ImageNet, is fine-tuned to classify a more specific dataset.

## Aim

- To understand the concepts of **Transfer Learning** and **ResNet** model.
- To build a **Transfer Learning** model for image classification in **PyTorch**.

## Tech Stack

- **Language**: Python
- **Libraries**:
  - **PyTorch**: For building and training the deep learning model.
  - **Pandas**: For data manipulation and handling.
  - **Matplotlib**: For visualizations.
  - **NumPy**: For numerical operations.
  - **OpenCV Python Headless**: For image processing (including resizing, loading, and manipulating image data).
  - **TorchVision**: For pre-trained models like **ResNet** and applying image transformations.

## Data Description

The dataset consists of images in three categories:

1. **Driving Licenses**
2. **Social Security Cards**
3. **Others**

The images are of varying shapes and sizes, which are preprocessed before modeling to ensure they are compatible with the ResNet model. The preprocessing steps include resizing, normalization, and label encoding.

## Approach

### 1. Data Loading

- **Organizing the Dataset**: Images are organized into separate directories for each category (driving licenses, social security cards, others).
- **Loading the Data**: The images are loaded from the respective folders and split into **train** and **test** sets.

### 2. Data Preprocessing

- **Resizing and Scaling**: All images are resized to a fixed dimension (200x200 pixels) to ensure consistency in the input size.
- **Encoding the Class Labels**: The class labels (driving license, social security, others) are encoded into **one-hot vectors** for classification.
- **Converting Images to Tensors**: The images are converted into **PyTorch tensors**, a format that the model can process.

### 3. Model Building and Training

- **ResNet50 Model**: The **ResNet50** model is loaded from **TorchVision**. We use a pre-trained version of ResNet, which has been trained on the **ImageNet** dataset, and fine-tune it for the current task.
- **Model Architecture**: The final fully connected layers of the ResNet model are modified to output predictions for 3 classes instead of 1000 (as it was originally trained on ImageNet).
- **Transfer Learning**: The initial layers of the model are frozen (i.e., their weights are not updated during training), and only the final layers are fine-tuned to learn from the new dataset.

### 4. Training

- **Loss Function**: **Cross-Entropy Loss** is used, as it's the most common loss function for classification tasks.
- **Optimizer**: **Adam Optimizer** is used for efficient training and weight updates.
- **Training Process**: The model is trained over 10 epochs with a batch size of 8. The training loss is printed at the end of each epoch.

### 5. Evaluation

- **Model Accuracy**: After training, the model is evaluated on the test set, and the **accuracy** is calculated by comparing the predicted labels with the true labels.
- **Results**:
  - **Training Loss**: The loss steadily decreases over 10 epochs.
  - **Final Accuracy**: The model achieves an accuracy of **67.11%** on the test set.

## Results

### Training Loss per Epoch:

- Epoch 1: 0.9792
- Epoch 2: 0.6098
- Epoch 3: 0.5533
- Epoch 4: 0.5136
- Epoch 5: 0.4120
- Epoch 6: 0.3926
- Epoch 7: 0.3930
- Epoch 8: 0.4320
- Epoch 9: 0.3713
- Epoch 10: 0.3754

### Final Model Accuracy:

**67.11%**

## Folder Structure

```
MLPipeline/
│
├── CreateDataset.py      # Handles dataset loading and preprocessing
├── ResNet_Model.py       # Defines and modifies the ResNet model
├── Train.py              # Trains the model and performs evaluation
└── output/               # Contains model checkpoints (model.pt)
```

## Model Architecture

The model used in this project is **ResNet50**, a deep convolutional neural network designed to solve the vanishing gradient problem and train deeper networks effectively. The key modifications include:

1. **Pre-trained Model**: The ResNet50 model is initialized with weights from ImageNet and adapted for the new task.
2. **Modified Fully Connected Layers**: The final fully connected layers are replaced to output predictions for 3 classes (driving license, social security, others).

## How to Run the Code

1. Install the necessary libraries:
   ```bash
   pip install torch torchvision pandas matplotlib opencv-python-headless numpy
   ```
2. Clone or download the repository.
3. Place the dataset images in appropriate folders under the `Data/` directory.
4. Run the notebook or Python script to train the model and evaluate its performance.

## Conclusion

In this project, we successfully implemented a **Transfer Learning** model using the **ResNet50** architecture. The model was trained on a custom image dataset and achieved a **67.11% accuracy** on the test set. While this is a good starting point, there is room for improvement by training for more epochs, fine-tuning hyperparameters, or enhancing the dataset.

---

### **Further Improvements:**

- Train for more epochs to improve accuracy.
- Fine-tune hyperparameters such as the learning rate.
- Augment the dataset with more diverse images (e.g., rotations, flips).
- Experiment with different pre-trained models (e.g., ResNet101, VGG16).
