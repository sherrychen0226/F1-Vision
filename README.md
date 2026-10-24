# F1-Vision
Formula 1 car team and driver classification using deep learning

1. Project Overview

Inspired by my interest in Formula 1, I developed an image classification system to recognize racing cars from four teams: Ferrari, McLaren, Mercedes, and Red Bull.

The project explores two tasks:

Team Classification: Identifying which of the four teams a car belongs to.

Driver Classification: Identifying the driver within each team.

The project was developed using Python and PyTorch, with a Gradio interface for image-based predictions.

2. Methodology

Dataset

I collected approximately 750 images from racing screenshots, official car photographs, and other on-track images, covering different viewing angles and racing conditions.

Team Classification

I fine-tuned a pretrained ResNet18 convolutional neural network to distinguish between four F1 teams.

Driver Classification

Driver identification proved more challenging because distinguishing features, particularly car numbers, often occupied only a small region of an image.

To address this, I adopted a two-stage approach:

Identify the team using the four-class model.

Use a separate binary classifier to distinguish the two drivers within that team.

Driver classification experiments were conducted for Ferrari, Mercedes, and McLaren. The Red Bull driver classifier remains incomplete.

3. Results

The team classification model achieved approximately 97% validation accuracy during fine-tuning.

Driver classification results were less consistent, with some experiments reaching approximately 85% validation accuracy. Performance varied between teams and was affected by limited training data and the visibility of car numbers.

These results reflect validation performance on the collected dataset rather than performance on a fully independent external benchmark.

4. Challenges and Lessons Learned

Overfitting: Later training epochs sometimes reduced validation performance, highlighting the importance of monitoring model generalization.

Data augmentation: Some image transformations, such as flipping images horizontally, could interfere with driver-specific visual information. This motivated a more careful approach to preprocessing for driver classification.

Limited visual information: Small or obscured car numbers made it difficult to reliably distinguish drivers using whole-car images.

Model design: Separating team and driver classification made the problem easier to analyze and allowed different models to focus on different identification tasks.

5. Future Improvements

Expand and balance the driver image dataset.

Explore cropping or attention mechanisms focused on driver numbers.

Complete Red Bull driver classification.

Test the models on a larger, independent dataset.

6. Technologies

Python

PyTorch and Torchvision

ResNet18

Gradio

Pillow and Matplotlib

