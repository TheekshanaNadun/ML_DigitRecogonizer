🔍 Model Type: Convolutional Neural Network (CNN)
CNNs are a powerful type of neural network often used for image classification tasks. They’re especially effective for recognizing patterns, like those in images.

📊 Dataset: MNIST
The MNIST dataset is a popular beginner’s dataset for image recognition. It consists of 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels. Our model learned to recognize these digits and classify them into 10 categories.

💡 Objective: Digit Recognition
The model’s purpose is to classify handwritten digits. This task has numerous applications, including digitizing written content and enhancing accessibility features.

⚙️ Hardware Comparison: CPU vs. GPU
I ran the training on both CPU and GPU to see the speed difference:

CPU training time per epoch: ~31 seconds

GPU training time per epoch (NVIDIA GeForce RTX 3050): ~12 seconds
The GPU provided a huge speed boost, cutting training time by over 50%!


🔍 Training Details:

Learning Rate: 0.001

Number of Epochs: 5

Test Accuracy: 98.96% 🎉
The model achieved a high accuracy on the test set, showing that it successfully learned to classify the digits!


📈 Evaluation Metrics:

Loss: Gradually decreased across epochs, indicating effective learning.

Accuracy: 98.96% — a great result for this type of task.


🔧 Key Takeaways:

1. Hardware Matters: GPUs significantly improve training speed, especially on larger datasets or more complex models.


2. Hyperparameters: The learning rate (set to 0.001) plays a crucial role in how fast and effectively the model learns.


3. Data Preparation: Using normalized data with transforms helped the model converge faster.
