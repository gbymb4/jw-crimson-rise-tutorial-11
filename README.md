# Deep Learning with PyTorch - Session 11: Training Stability & Regularization Fundamentals

**Objective:**    
Learn core regularization techniques (dropout, L1/L2) and training stability methods (learning rate schedulers, early stopping) through hands-on implementation and practice.

---

## Session Timeline (1 Hour)

| Time      | Activity                                       |
| --------- | ---------------------------------------------- |
| 0:00-0:05 | 1. Check-in + Recap version control session   |
| 0:05-0:30 | 2. Guided Example: Regularization in Action   |
| 0:30-0:55 | 3. Solo Exercise: Implement Your Own Setup    |
| 0:55-1:00 | 4. Quick Review & Next Session Preview        |

---

## Session Steps & Instructions

### 1. **Check-in & Recap** (5 min)
- Welcome and attendance
- Quick review: How did the GitHub workflow feel? Any issues with commits/pushes?
- Today's focus: Making training more stable and preventing overfitting

---

### 2. **Guided Example: Regularization Demo** (25 min)

#### **Concepts Introduction** (5 min)
- **Overfitting**: Model memorizes training data, poor generalization
- **Regularization**: Techniques to improve generalization
- **Training Stability**: Ensuring consistent, reliable training

#### **Live Coding Demo** (20 min)
Work through `regularization_demo.py` together:

**Key Components to Demonstrate:**
1. **Baseline Model** - Simple network without regularization
2. **Dropout** - Random neuron deactivation during training
3. **L1/L2 Regularization** - Weight penalty in loss function
4. **Learning Rate Scheduler** - Reduce LR when plateau detected
5. **Early Stopping** - Stop when validation loss stops improving

**Script Structure:**
```python
# 1. Load CIFAR-10 (more complex than MNIST)
# 2. Define three model variants:
#    - BaselineNet (no regularization)
#    - RegularizedNet (dropout + L2)
#    - ScheduledNet (+ LR scheduler + early stopping)
# 3. Training function with validation tracking
# 4. Compare all three models side-by-side
# 5. Plot training curves showing the differences
```

**Live Discussion Points:**
- When to use dropout (training vs evaluation mode)
- How L1 vs L2 regularization affects weights differently
- Why learning rate scheduling helps convergence
- Early stopping criteria and patience parameters

---

### 3. **Solo Exercise: Build Your Regularization Toolkit** (25 min)

Students work on `regularization_exercise.py` with the following structure:

#### **Exercise Requirements:**
1. **Dataset**: Fashion-MNIST (10 clothing categories)
2. **Base Model**: 3-layer fully connected network
3. **Your Tasks**:
   - Implement dropout layers with configurable rates
   - Add L2 regularization to the optimizer
   - Create a simple learning rate scheduler
   - Implement basic early stopping logic
   - Train and compare 2 variants: baseline vs regularized

#### **TODO Structure:**
```python
# TODO 1: Add dropout layers to the network
# TODO 2: Configure L2 regularization in optimizer
# TODO 3: Implement ReduceLROnPlateau scheduler
# TODO 4: Add early stopping with patience=5
# TODO 5: Plot comparison of training/validation curves
# TODO 6: Calculate and compare final test accuracies
```

#### **Expected Outcomes:**
- See overfitting in baseline model
- Observe regularization effects on training curves
- Experience hands-on implementation of key techniques

---

### 4. **Review & Preview** (5 min)
- Quick sharing: What patterns did you notice in your training curves?
- Key takeaways: Regularization trade-offs, when each technique helps
- Next session preview: Advanced regularization (BatchNorm, gradient clipping) and training diagnostics

---

## **Supporting Materials Needed**

### **Instructor Script: `regularization_demo.py`**
- Complete working example with CIFAR-10
- Three model variants with clear comparisons
- Visualization code for training curves
- Comments explaining each regularization technique

### **Student Template: `regularization_exercise.py`**
- Fashion-MNIST data loading (pre-written)
- Basic network structure with TODO placeholders
- Training loop framework with validation tracking
- Plotting template ready for completion

### **Quick Reference Sheet**
- Dropout: When to use .train() vs .eval()
- L1/L2 regularization syntax in PyTorch optimizers
- Common learning rate scheduler options
- Early stopping implementation patterns

---

## **Session Learning Goals**
By the end of this session, students should:
1. Understand why regularization prevents overfitting
2. Know how to implement dropout, L1/L2, and basic schedulers
3. Recognize overfitting patterns in training curves
4. Have working code templates for future projects
5. Appreciate the iterative nature of ML experimentation
