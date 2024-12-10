# maltese-or-poodle
Implementation of a neural network using transfer learning to classify between maltese and poodle dog breeds

# Dog Breed Classifier: Maltese or Poodle (and Generalized Approach)

This project employs **transfer learning** to classify dog breeds, leveraging insights from related projects such as the **ML4A Transfer Learning Guide** and the **Cats vs. Dogs Transfer Learning Notebook**. Both resources contributed practical examples of reusing pre-trained models and optimizing for new datasets.

## Overview

### Initial Plan
The initial plan was to create a classifier distinguishing between **Maltese** and **Poodle** breeds. However, due to the breeds' visual similarity, a second dataset using more distinct breeds, **Pug** and **Whippet**, was adopted for comparison and optimization.

### Insights from Reference Notebooks
1. **ML4A Transfer Learning Notebook**:
   - Introduced reusable pipelines for transfer learning using pre-trained models.
   - Highlighted visualization of feature activations to interpret model decisions.
   - Applied the **InceptionV3** architecture as an example of a pre-trained network.
2. **Cats vs. Dogs Transfer Learning Notebook**:
   - Focused on splitting datasets manually and using TensorFlow for data augmentation.
   - Demonstrated techniques to reduce overfitting through regularization and dropout.

### Final Approach
This project synthesizes the best practices from the reference notebooks with the following key elements:
- **MobileNet** and **VGG16** architectures as base models.
- Extensive use of data augmentation and normalization to improve generalization.
- Comparison of results using two sets of breeds: visually similar (Maltese and Poodle) and distinct (Pug and Whippet).

## Features

### General Workflow
1. **Dataset Handling**:
   - Images are structured in directories for training and testing.
   - Datasets are split using TensorFlow or manually for more control.
2. **Model Architectures**:
   - **MobileNet**: Lightweight and optimized for faster training.
   - **VGG16**: Robust architecture for high-capacity learning.
3. **Data Augmentation**:
   - Techniques like random flipping, zoom, and rotation to create diverse training samples.
4. **Regularization**:
   - Dropout layers and freezing pre-trained layers for reduced overfitting.
5. **Loss and Metrics**:
   - Binary or categorical crossentropy depending on the dataset.
   - Accuracy as the primary metric for performance evaluation.

### MobileNet Implementation
Inspired by both references:
- Data augmentation pipeline integrated with normalization layers.
- Training for **70 epochs** to optimize MobileNet's learning capacity.
- Augmented dataset improves model generalization.

### VGG16 Implementation
Following the Cats vs. Dogs Transfer Notebook:
- Pre-trained VGG16 used with `include_top=True` for the classification task.
- Fine-tuning by freezing all but the last layers.
- Validation and test performance plotted to visualize overfitting risks.

## Results

### Key Comparisons
1. **Training Accuracy**:
   - MobileNet: Peaks at ~87% after 70 epochs.
   - VGG16: Achieves ~85% in fewer epochs but with higher capacity.
2. **Validation Accuracy**:
   - Similar trends observed with both datasets, confirming the effectiveness of transfer learning.
3. **Impact of Data Augmentation**:
   - Significantly improves validation performance, especially for distinct breeds.

### Additional Visualizations
- Feature activation maps for both architectures (based on ML4A guide).
- Training/validation accuracy and loss trends for each model and dataset.

## How to Run
1. Run the Colab notebooks:
   - For MobileNet: `maltese-or-poodle.ipynb`
   - For VGG16: `transfer-learning-dog-breeds.ipynb`
  
## References

### Code Notebooks
1. [ML4A Transfer Learning Guide](https://colab.research.google.com/github/kylemath/ml4a-guides/blob/master/notebooks/transfer-learning.ipynb#scrollTo=rFL-fLitYoa3)
2. [Cats vs Dogs Transfer Learning](https://colab.research.google.com/github/cunhamaicon/catsxdogs/blob/master/catsxdogs_transfer.ipynb#scrollTo=9bKzjWBbnXaM)

### Dataset
- [Maltese and Poodle Dataset](https://github.com/gskumlehn/maltese-or-poodle)

