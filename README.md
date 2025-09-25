# DL-september_exam
Deep Learning model (MLP) to predict whether a user clicked on an advertisement or not.
Model choice

Architecture: a fully-connected feedforward neural network (MLP) with a small embedding branch for Country.
Why: data are tabular / mixed-type (numerical + categorical + high-dim sparse binary vector). MLPs with embeddings handle this efficiently and were covered in the course; convolutional / recurrent models bring no advantage here.

INPUT — preprocessing & representation

a) Preprocessing (short steps):

Numerical features (Daily Time Spent, Age, Area Income, Daily Internet Usage) → standardize (zero mean, unit variance) with StandardScaler.
Binary features (Male, Search Queries 300-dim) → keep as 0/1 floats; no scaling required for binary but convert to float32. Consider sparse representation if memory is an issue.
Country (235 categories) → learned embedding (see below). Encode with integer IDs using OrdinalEncoder or LabelEncoder.
Timestamp → extract hour, day_of_week, month, and is_weekend. Represent cyclical features (hour, month, day_of_week) as sin/cos pairs to preserve circularity. Then standardize continuous derived features.
Missing values → impute (median for numeric; mode/“unknown” for categorical).

b) Model input shape & domain (after preprocessing):

Suggested representation:
Numeric (std): 5 features (DailyTime, Age, AreaIncome, DailyInternetUsage, Male) → shape (batch, 5).
Timestamp-derived: 4 (e.g., sin_hour, cos_hour, sin_dow, cos_dow) → (batch,4).
Country embedding: embedding dim E = 16 → produced by an embedding lookup giving (batch,16).
Search queries: 300 binary → (batch,300).
Concatenate into a dense input vector of size 5 + 4 + 16 + 300 = 325.
All values are float32; model input shape: (batch_size, 325); values mostly in ~[-3, +3] after standardization for numeric, and 0/1 for search bits.

OUTPUT and LOSS

Output layer: a single neuron with sigmoid activation producing probability ∈(0,1). Implementation best practice: produce a single scalar logit and use a numerically stable loss.

Loss function: Binary Cross-Entropy with logits (BCEWithLogitsLoss in PyTorch).
Why: binary classification; BCE is the canonical probabilistic loss that optimizes log-likelihood for Bernoulli labels. If class imbalance becomes an issue, use pos_weight or focal loss; here classes are roughly balanced (~477/953), so vanilla BCE is fine.

MODEL CONFIGURATION

a) Overall pipeline (textual/ASCII figure):

[Numeric (5)] \
[Timestamp (4)] ---> [Concatenate] -> [Dense 512 → BN → ReLU → Dropout]                                     -> [Dense 256 → BN → ReLU → Dropout]
[Country Emb (16)] /                 -> [Dense 64  → BN → ReLU → Dropout] -> [Linear 1 (logit)]                                     
[Search Queries (300)] /

(Optionally: first pass the 300-dim Search Queries through a 1-2 layer bottleneck like Dense 256 -> ReLU -> Dense 128 before concatenation to learn interactions.)

b) Hyperparameters & techniques to optimize performance:

Layers: 3 dense layers (512 → 256 → 64) with BatchNorm, ReLU and Dropout (0.3–0.5).
Optimizer: AdamW (or Adam) with lr = 1e-3 and weight_decay = 1e-5.
Batch size: 32–128 (e.g., 64).
LR schedule: ReduceLROnPlateau or Cosine annealing with warm restarts.
Regularization: dropout, L2 weight decay, early stopping (patience 5 epochs on val loss).
Class handling: compute class weights if severe imbalance; use weighted BCE.
Feature interactions: allow network to learn them; optionally add pairwise cross features or an attention layer for search-query importance.
Initialization & seeds: set random seeds and deterministic PyTorch settings for reproducibility.
Validation during training: keep a stratified validation split (e.g., 80/20 train/val) or use cross-validation for robust tuning.
Monitoring: log training/val loss, ROC-AUC, PR-AUC, F1 with TensorBoard or wandb.

MODEL EVALUATION

How to assess generalization (recommended setup):
Hold-out test set: split dataset into train (70%) / val (15%) / test (15%), using stratified splits on the target to preserve click ratio. Keep test untouched until final evaluation.
Hyperparameter tuning: run Stratified K-Fold CV (e.g., 5-fold) inside training for robust estimates and then confirm best model on hold-out test. Use nested CV if you want unbiased hyperparam selection estimates.
Metrics: report ROC-AUC, PR-AUC, Accuracy, F1, Precision, Recall, and Calibration (Brier score). If business cares about positive clicks precision, report precision@k or expected lift.
Statistical stability: report mean ± std of metrics across folds; use bootstrap CI on test metrics for uncertainty.
Calibration check: reliability plots; consider temperature scaling or Platt scaling if probabilities are poorly calibrated.
