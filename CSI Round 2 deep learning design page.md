Drone vs. Bird Detection System - Design Document

BY: S Kausik Ragav

Problem Statement

The objective was to design an automated defense system capable of
distinguishing between drones and other aerial objects (birds) using a
Deep Learning approach. The main challenge is differentiating between
two small, flying objects that often share similar bodies or figures
against complex sky backgrounds.

Data Pipeline Design

To ensure efficient training on the provided dataset, I implemented the
following preprocessing steps:

Resolution: I chose an image size of (100, 100). This is a trade-off; it
is large enough to capture the rotor details of a drone but small enough
to allow for a larger batch size (32) on limited hardware (8GB RAM
laptop).

Normalization: Pixel values were rescaled from \[0, 255\] to \[0, 1\]
using a \`Rescaling\` layer. This helps the gradient descent converge
faster.

Performance: I used AUTOTUNE with .cache() and .prefetch(). This ensures
the GPU/CPU is never waiting for data to be loaded from the disk,
effectively parallelizing data loading and model training.

Model Architecture (CNN)

I designed a Convolutional Neural Network (CNN) with three main feature
extraction blocks. I chose to make my own CNN over a massive pre-trained
model (like ResNet) to keep the model lightweight for potential
deployment (e.g., running on the drone/camera hardware itself).

Conv2D Layers I used increasing filter sizes (32, 64, 128) to capture
low-level features (edges) first and high-level features
(shapes/structures) later.

Batch Normalization: Added after every convolution. This was crucial for
stabilizing the learning process and preventing the \"vanishing
gradient\" problem.

Dropout (0.4): A relatively aggressive dropout rate of 40% was placed
before the final classification layer. Since the dataset might have
similar-looking images, this forces the model to learn robust features
rather than memorizing specific pixel patterns (overfitting).

Output:A single neuron with a \`sigmoid\` activation, ideal for binary
classification (0 = Bird, 1 = Drone).

Experiment Tracking (Weights & Biases)

I integrated \`wandb\` to track the training loss and validation
accuracy in real-time. This allowed me to monitor for overfitting (where
training loss goes down, but validation loss goes up).

Ablation Study

To understand the impact of specific architectural choices, I conducted
a brief ablation study comparing the full model against a simplified
version.

Experiment A (Baseline):Full model with Batch Normalization.

Experiment B (Ablation):Removed \`BatchNormalization\` layers.

Observation:The model without Batch Normalization learned significantly
slower and had higher variance in validation accuracy during the first 3
epochs. This confirmed that Batch Norm is essential for this specific
dataset configuration. During the ablation study, I reduced the training
duration to 3 epochs for speed. I observed that the model failed to
generalize, misclassifying a clear image of a drone as a bird with 99%
confidence. This confirms that 3 epochs is insufficient for feature
convergence, and a longer training schedule (15+ epochs) is required for
deployment.

![](media/image1.png){width="6.268055555555556in"
height="3.015277777777778in"}

Conclusion

The final model achieves a balance between inference speed and accuracy.
The use of a custom lightweight CNN makes it suitable for real-time
defense applications where low latency is critical.
