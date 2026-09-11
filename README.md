# CNN Adversarial Robustness Evaluation

## Project Overview

This project evaluates the robustness of Convolutional Neural Networks against adversarial perturbations using CIFAR-10. It demonstrates that standard CNNs achieve high accuracy on clean data but collapse under small, imperceptible perturbations, raising critical concerns for deployment in security-sensitive applications.

**Key Finding:** Model accuracy drops from 71.2% to 11.8% with imperceptible noise (ε=0.05).

## Problem Statement

How robust are CNNs to adversarial perturbations, and how does perturbation magnitude affect classification accuracy? Which image classes are most vulnerable?

## Dataset

**CIFAR-10:** 60,000 labeled 32×32 RGB images across 10 object classes
- Training: 50,000 images (40,000 train, 10,000 validation)
- Test: 10,000 images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Project Structure

```
ROBUSTNESS-OF-CONVOLUTIONAL-NEURAL-NETWORKS-UNDER-ADVERSARIAL-PERTURBATIONS/
├── 01_Data_Loading_and_EDA.ipynb          # Dataset exploration & visualization
├── 02_Data_Preprocessing.ipynb            # Normalization & train/val/test split
├── 03_CNN_Model.ipynb                     # Architecture design
├── 04_Training_and_Evaluation.ipynb       # Model training & clean accuracy
├── 05_Adversarial_Perturbations.ipynb     # Adversarial robustness testing
├── 06_Robustness_Evaluation.ipynb         # Final analysis & metrics
├── data/
│   └── processed/
│       └── cifar10_preprocessed.pkl
├── models/
│   ├── cnn_architecture.keras
│   └── cnn_trained.keras
├── figures/
│   ├── 01_sample_images.png
│   ├── 02_class_distribution.png
│   ├── 03_pixel_distribution.png
│   ├── 04_brightness_analysis.png
│   ├── 04_learning_curves.png
│   ├── 04_confusion_matrix.png
│   ├── 04_per_class_accuracy.png
│   ├── 05_adversarial_examples.png
│   ├── 05_brightness_robustness_hypothesis.png
│   ├── 05_per_class_robustness.png
│   ├── 05_robustness_curve.png
│   └── 06_robustness_evaluation_summary.png
├── results/
│   ├── training_results.pkl
│   ├── adversarial_results.pkl
│   ├── robustness_summary.csv
│   └── per_class_robustness.csv
├── README.md
└── requirements.txt
```

## Key Findings

### 1. Hidden Fragility
Standard CNNs achieve 71.2% clean accuracy but collapse under adversarial perturbations. The model learns precise but brittle features vulnerable to small noise.

### 2. Non-Linear Degradation
- ε = 0.0: 71.2% accuracy (baseline)
- ε = 0.05 (imperceptible): 11.8% accuracy (59.4% drop)
- ε = 0.20 (noticeable): 8.4% accuracy (plateau)

### 3. Brightness Predicts Vulnerability
Bright classes (ship: 86.9%, airplane: 75.8%) are 2-3× more robust than dark classes (cat: 50.7%, frog: 85.7%). Correlation: r = 0.62.

**Why?** Higher signal-to-noise ratio in bright images preserves discriminative features under perturbation.

### 4. Class Confusion Patterns
- Cats ↔ Dogs: 199 confusions (semantic similarity)
- Birds ↔ Airplanes: 125 confusions
- Deer ↔ Horses: 66 confusions

### 5. Accuracy-Robustness Tradeoff
High clean accuracy does NOT guarantee robustness. This is a fundamental property of neural networks trained on clean data.

## Model Architecture

**Convolutional Neural Network:**
- Conv2D(32, 3×3) → MaxPool(2×2) → Dropout(0.25)
- Conv2D(64, 3×3) → MaxPool(2×2) → Dropout(0.25)
- Conv2D(128, 3×3) → Dropout(0.25)
- Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(10, softmax)

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse categorical crossentropy
- Batch size: 128
- Early stopping: patience=10

**Results:**
- Clean test accuracy: 71.20%
- Training epochs: 47 (early stopped)

## Adversarial Evaluation

**Perturbation Method:** Gaussian noise at epsilon values {0.0, 0.01, 0.05, 0.10, 0.20, 0.30}

**Robustness Metrics:**
- Accuracy drops per epsilon level
- Per-class vulnerability ranking
- Brightness-robustness correlation
- Critical perturbation threshold

**Finding:** Model becomes unreliable (< 50% accuracy) at ε = 0.10

## Practical Implications

### Suitable Deployments
- Academic benchmarking
- Educational demonstrations
- Baseline comparisons
- Research environments

### Unsuitable Deployments
- Security systems (surveillance, access control)
- Autonomous vehicles
- Medical diagnostics
- Any application where adversaries manipulate inputs

## Improving Robustness

1. **Adversarial Training:** Train on clean + perturbed examples (5-10% accuracy cost)
2. **Certified Defenses:** Formal robustness guarantees (e.g., randomized smoothing)
3. **Ensemble Methods:** Combine multiple models
4. **Data Augmentation:** Noise injection during training
5. **Architecture Innovation:** Vision Transformers show better robustness

## Requirements

```
Python 3.8+
TensorFlow 2.10+
NumPy 1.21+
Matplotlib 3.5+
Seaborn 0.11+
Scikit-learn 1.0+
Pandas 1.3+
```

Install with:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pandas
```

## Usage

Run notebooks in order:
```bash
# 1. Explore data
jupyter notebook 01_Data_Loading_and_EDA.ipynb

# 2. Preprocess
jupyter notebook 02_Data_Preprocessing.ipynb

# 3. Design model
jupyter notebook 03_CNN_Model.ipynb

# 4. Train model
jupyter notebook 04_Training_and_Evaluation.ipynb

# 5. Test robustness
jupyter notebook 05_Adversarial_Perturbations.ipynb

# 6. Analyze results
jupyter notebook 06_Robustness_Evaluation.ipynb
```

## Key Code Example

```python
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('models/cnn_trained.keras')

# Add adversarial perturbation
epsilon = 0.05
x_perturbed = np.clip(x_test + epsilon * np.random.normal(0, 1, x_test.shape), 0, 1)

# Evaluate robustness
y_pred = model.predict(x_perturbed)
accuracy = (y_pred.argmax(axis=1) == y_test).mean()
print(f"Accuracy under perturbation: {accuracy:.2%}")
```

## Results Summary

| Metric | Value |
|--------|-------|
| Clean Accuracy | 71.20% |
| Accuracy at ε=0.05 | 11.80% |
| Accuracy at ε=0.20 | 8.40% |
| Accuracy Drop (ε=0.05) | 59.40% |
| Most Robust Class | Ship (86.9%) |
| Most Vulnerable Class | Cat (50.7%) |
| Brightness-Robustness Correlation | 0.62 |

## Conclusions

This evaluation demonstrates that standard CNNs are fundamentally vulnerable to adversarial perturbations. The dramatic accuracy collapse with imperceptible noise (71.2% → 11.8%) is not unique to this model but represents a core challenge in deep learning.

**Critical Insight:** High accuracy on clean benchmarks provides a false sense of security. Robustness must be explicitly addressed during model design and training for deployment in security-critical applications.

## References

[1] Krizhevsky, A. "Learning Multiple Layers of Features from Tiny Images." University of Toronto, 2009.

[2] Goodfellow, I. J., Shlens, J., Szegedy, C. "Explaining and Harnessing Adversarial Examples." arXiv:1412.6572, 2014.

[3] TensorFlow Documentation. https://tensorflow.org

## Author

**Gifty Acquah**  
PhD Candidate | Agentic AI Safety | Trustworthy Multi-Agent Systems | Critical Infrastructure Security
Concordia University 
Email: giftyacquah999@gmail.com

## License

This project is for educational purposes as part of the Code First Girls +Masters program.

---

**Last Updated:** August 2026  
**Status:** Complete (Assignments 1 & 2)
