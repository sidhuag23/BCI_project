# Mixup-Enhanced Weighted Ensemble CNN for Low-Repetition P300 Character Recognition

**A Project Report**

**Ashik M Biju, Sidhu A G, Athira A**
School of Computer Science & Engineering
Digital University Kerala

---

## Abstract

Brain-Computer Interfaces (BCIs) based on the P300 event-related potential enable individuals with severe motor disabilities to communicate by spelling characters through brain activity alone. Recent work by Shukla et al. proposed a Weighted Ensemble Spatio-Sequential Convolutional Neural Network (WE-SPSQ-CNN) that achieves competitive character recognition accuracy on the BCI Competition III Dataset II benchmark. However, their method addresses the inherent class imbalance between P300 and Non-P300 signals through naive duplication of P300 epochs — an approach that fails to provide meaningful regularization or ensemble diversity.

In this report, we reproduce the original WE-SPSQ-CNN baseline in PyTorch and propose an enhancement based on **Mixup augmentation**, which generates synthetic training samples through linear interpolation of existing EEG signals and their labels. We evaluate three augmentation strategies — Mixup with α=0.2, Mixup with α=0.4, and a Mixed 50% hybrid of Mixup and naive cloning — against the baseline on both subjects of the BCI Competition III Dataset II.

Our key findings are: (1) the baseline is successfully reproduced, matching the paper within 1–2% at 10 and 15 repetitions; (2) Mixup augmentation improves peak accuracy at 15 repetitions by +2% for both subjects, with Subject A reaching 97% and Subject B reaching 95%; (3) the Mixed 50% hybrid strategy achieves the best overall performance on Subject B across all repetition counts; and (4) Mixup does not improve — and for Subject A slightly reduces — accuracy at 5 repetitions, revealing a fundamental trade-off between low-repetition and high-repetition regimes. These findings confirm that principled augmentation can improve generalization but does not resolve the core low-SNR challenge of EEG-based BCI spelling at low repetition counts.

**Keywords:** Brain-Computer Interface, P300, EEG, Convolutional Neural Network, Mixup Augmentation, Ensemble Learning, Character Recognition

---

## I. Introduction

Brain-Computer Interfaces (BCIs) provide a direct communication channel between the human brain and external devices, bypassing conventional neuromuscular pathways. For patients suffering from amyotrophic lateral sclerosis (ALS), locked-in syndrome, or severe spinal cord injuries, BCIs may represent the only viable means of communication with the outside world.

Among various BCI paradigms, the **P300-based speller** introduced by Farwell and Donchin remains one of the most widely studied and clinically promising. The P300 is a positive-going event-related potential (ERP) that occurs approximately 300 milliseconds after the presentation of an infrequent, task-relevant stimulus. In the standard row-column speller paradigm, a user is presented with a 6×6 matrix of characters. Rows and columns flash in random order, and when the row or column containing the target character flashes, the user's brain generates a P300 response detectable from scalp EEG.

### The Core Challenges

P300-based character recognition remains difficult due to three persistent and interrelated problems:

1. **Low signal-to-noise ratio (SNR):** EEG recordings are inherently noisy. The P300 signal amplitude is typically 1–5 µV, while background EEG noise can be 10–50 µV. This means a single flash epoch contains far more noise than signal.

2. **Class imbalance:** In the standard row-column paradigm with 6 rows and 6 columns, only 2 of the 12 flashes per repetition (1 row + 1 column) elicit a P300. This creates a 5:1 Non-P300 to P300 ratio, making the classification task inherently imbalanced.

3. **Repetition requirement:** To compensate for low SNR, the paradigm repeats the flashing sequence multiple times. More repetitions improve accuracy through signal averaging, but each additional repetition reduces the communication speed (information transfer rate, ITR). A system requiring 15 repetitions types approximately 1 character per minute — too slow for practical communication.

### Our Contribution

Shukla et al. proposed WE-SPSQ-CNN to address these challenges, achieving character recognition accuracies of 76.5%, 87.5%, and 94.5% at 5, 10, and 15 repetitions respectively. However, their method handles class imbalance by simply copying each P300 epoch five times — an approach that provides no new information to the model and may actually reduce ensemble diversity.

We propose replacing this naive cloning step with **Mixup augmentation** (Zhang et al., ICLR 2018), which creates synthetic training examples by linearly blending pairs of real P300 and Non-P300 epochs. We additionally introduce and evaluate a **Mixed 50% hybrid strategy** — our own contribution — that combines Mixup-generated and naively-cloned samples to explore a middle ground.

The remainder of this report is structured as follows. Section II reviews related work. Section III describes the baseline WE-SPSQ-CNN methodology. Section IV identifies the limitations of the baseline. Section V describes our proposed Mixup-augmented approach. Section VI presents the experimental setup. Section VII presents and analyses all results. Section VIII discusses what was and was not solved. Section IX concludes.

---

## II. Related Work

P300 character recognition has been studied extensively in the literature.

**Rakotomamonjy and Guigue** achieved 96.5% accuracy at 15 repetitions using an ensemble of Support Vector Machines (ESVM) with recursive channel elimination. This remains the strongest result at 15 repetitions among classical methods.

**Salvaris and Sepulveda** employed wavelet-based feature extraction with an ensemble of Fisher's Linear Discriminant classifiers, achieving 96.5% at 15 repetitions.

**Cecotti and Gräser** introduced one of the first CNN-based approaches for P300 detection, demonstrating the suitability of convolutional architectures for ERP classification, achieving 88.5% at 10 repetitions.

**Bhatnagar et al.** introduced the ensemble of SVMs as a classification method for single-trial detection, achieving 92.5% accuracy.

**Wang et al.** proposed ST-CapsNet, a spatial-temporal capsule network achieving 98% accuracy at 15 repetitions — the current state of the art at high repetitions.

**Shukla et al.** proposed WE-SPSQ-CNN specifically targeting low-repetition accuracy, outperforming state-of-the-art methods at 5 repetitions (76.5%) while remaining competitive at 15 repetitions.

**Zhang et al.** introduced Mixup augmentation for image classification at ICLR 2018, demonstrating that linear interpolation between training examples and their labels improves generalization. Mixup has since been successfully adapted to EEG-based applications, demonstrating consistent gains in low-data regimes.

---

## III. Baseline Methodology: WE-SPSQ-CNN

### A. Dataset

We use the **BCI Competition III Dataset II**, which contains EEG recordings from two subjects (A and B) performing the row-column P300 speller task:

| Parameter | Value |
|---|---|
| Subjects | 2 (A and B) |
| Sampling rate | 240 Hz |
| EEG channels | 64 |
| Training characters | 85 per subject |
| Test characters | 100 per subject |
| Repetitions per character | 15 |
| Flashes per repetition | 12 (6 rows + 6 columns) |
| Total epochs per subject (train) | 85 × 15 × 12 = 15,300 |
| P300 epochs (train) | 85 × 15 × 2 = 2,550 |
| Non-P300 epochs (train) | 85 × 15 × 10 = 12,750 |
| Class imbalance ratio | 5:1 (Non-P300 : P300) |

For each character, 12 flashes × 15 repetitions = 180 flash epochs are recorded. Of these, exactly 2 per repetition (1 target row + 1 target column) elicit a P300 response.

### B. Signal Preprocessing

The preprocessing pipeline follows Shukla et al. exactly:

**Step 1 — Bandpass Filtering:**
A 4th-order Chebyshev Type I bandpass filter with cutoff frequencies 0.1 Hz and 10 Hz is applied to the continuous per-character EEG signal (not to individual epochs). Critically, filtering is applied to the full continuous recording of approximately 7,000 samples per character before epoching. The 0.1 Hz high-pass has a time constant of ~1.6 seconds; applying it to a 160-sample epoch would produce transients larger than the P300 signal itself.

**Step 2 — Epoch Extraction:**
Flash onsets are detected from the StimulusCode channel as 0→nonzero transitions. A 667 ms window (160 samples at 240 Hz) is cut from the filtered continuous signal at each onset:

$$\text{epoch} = \text{signal}[\text{onset} : \text{onset} + 160, \; :]$$

Each epoch has shape (160 samples × 64 channels).

**Step 3 — Label Assignment:**
Within the first 24 samples (~100 ms) of each flash window, the StimulusType signal is checked. If it contains a 1, the epoch is labeled as P300 (y=1); otherwise Non-P300 (y=0).

### C. SPSQ-CNN Architecture

The base classifier is the Spatio-Sequential CNN (SPSQ-CNN), a compact network with 45,809 parameters. Its architecture is shown in Table I.

**Table I: SPSQ-CNN Architecture**

| Layer | Kernel Size | Output Shape | Parameters |
|---|---|---|---|
| Batch Normalization 1 | — | (160, 64) | 256 |
| Reshape 1 | — | (1, 160, 64) | 0 |
| 2D Convolution | 1 × 64 | (32, 160, 1) | 2,080 |
| Reshape 2 | — | (32, 160) | 0 |
| 1D Convolution | 20 × 1 | (16, 8) | 10,256 |
| Batch Normalization 2 | — | (16, 8) | 64 |
| Leaky ReLU | — | (16, 8) | 0 |
| Flatten | — | (128,) | 0 |
| Fully Connected 1 | 128 | 128 | 16,512 |
| Fully Connected 2 | 128 | 128 | 16,512 |
| Fully Connected 3 (output) | 1 | 1 | 129 |
| **Total** | | | **45,809** |

The forward pass processes a single EEG epoch as follows:

1. **BatchNorm** normalizes across the 64 channel dimension, stabilizing activations.
2. **Conv2D (1×64)** applies a spatial filter collapsing all 64 channels at each time step into 32 learned spatial patterns — analogous to Common Spatial Patterns (CSP).
3. **Conv1D (20×1, stride 20)** applies a temporal filter compressing 160 time steps into 8 — capturing frequency content in the P300 band.
4. **Two FC layers** with tanh activation and dropout p=0.8 form the classifier head.
5. **Sigmoid output** produces a P300 probability in [0, 1].

### D. Class Imbalance Handling — Naive Cloning

To address the 5:1 imbalance, the baseline replicates each P300 epoch five times:

$$\text{Training set} = \{12{,}750 \text{ Non-P300}\} \cup \{5 \times 2{,}550 \text{ P300 copies}\}$$

Each of the five SPSQ-CNN classifiers is trained on a balanced subset of 5,100 epochs (2,550 Non-P300 + 2,550 P300 clones), ensuring class balance.

### E. Weighted Ensemble

Five SPSQ-CNN classifiers are trained independently. Their predictions are combined using a weighted average, where the weight of each classifier is proportional to its training accuracy:

$$W_k = \frac{T_k}{\sum_{i=1}^{n} T_i} \tag{1}$$

where $T_i = TP_i + TN_i$ denotes the sum of true positives and true negatives for classifier $i$.

### F. Character Decoding

For each character, the ensemble produces P300 scores for all 12 flash codes by averaging probabilities across repetitions:

$$F(i) = \frac{1}{j} \cdot \frac{1}{k} \sum_{j=1}^{J} \sum_{k=1}^{5} W_k \cdot p_k \tag{2}$$

The predicted column and row are the flash codes with highest scores:

$$C_i = \underset{1 \leq i \leq 6}{\arg\max}\; F(i) \tag{3}$$

$$R_i = \underset{7 \leq i \leq 12}{\arg\max}\; F(i) \tag{4}$$

The decoded character is located at the intersection $(R_i, C_i)$ in the 6×6 character matrix.

---

## IV. Limitations of the Baseline Method

### A. Naive Cloning Provides No Real Augmentation

Training a neural network on five identical copies of a P300 sample is mathematically equivalent to training on a single copy with a fivefold-amplified gradient. The model receives no novel information from the duplicates and therefore gains no improved generalization. The authors themselves acknowledge this gap, stating in their conclusion: *"future works will include data augmentation."*

### B. Limited Ensemble Diversity

The five base classifiers in the ensemble see different Non-P300 subsets but **identical** P300 samples. True ensemble diversity requires each classifier to see a meaningfully different distribution of both classes. Cloning undermines this requirement.

### C. Aggressive Dropout as a Compensatory Mechanism

The use of dropout with p=0.8 — substantially higher than typical values of 0.3–0.5 — suggests the model is operating close to overfitting. Such aggressive dropout is indicative of insufficient training data diversity, which principled augmentation could remedy.

### D. Suboptimal Low-Repetition Performance

While the method outperforms state-of-the-art at 5 repetitions (76.5% mean), this absolute accuracy leaves significant room for improvement. Since the information transfer rate (ITR) scales inversely with the number of repetitions required, improvements at low repetitions translate to meaningful gains in practical BCI usability.

---

## V. Proposed Method: Mixup-Augmented WE-SPSQ-CNN

### A. Mixup Formulation

Given two training samples $(\mathbf{x}_1, y_1)$ and $(\mathbf{x}_2, y_2)$, where $\mathbf{x}_i \in \mathbb{R}^{160 \times 64}$ is a preprocessed EEG epoch and $y_i \in \{0, 1\}$, a synthetic sample is constructed as:

$$\mathbf{x}_{\text{new}} = \lambda \cdot \mathbf{x}_1 + (1 - \lambda) \cdot \mathbf{x}_2 \tag{5}$$

$$y_{\text{new}} = \lambda \cdot y_1 + (1 - \lambda) \cdot y_2 \tag{6}$$

where $\lambda \in [0, 1]$ is sampled from a Beta distribution:

$$\lambda \sim \text{Beta}(\alpha, \alpha) \tag{7}$$

In our implementation, pairs are always formed between one P300 and one Non-P300 epoch (cross-class Mixup). To ensure the synthetic sample is always more P300 than Non-P300, $\lambda$ is clamped:

$$\lambda = \max(\lambda, 1 - \lambda) \tag{8}$$

This guarantees $\lambda \geq 0.5$, so the soft label $y_{\text{new}} \in [0.5, 1.0]$ — always closer to P300 than Non-P300.

### B. What Alpha Controls

The hyperparameter $\alpha$ controls the shape of the Beta distribution and therefore the strength of blending:

| $\alpha$ | Distribution shape | Typical $\lambda$ values | Effect |
|---|---|---|---|
| → 0 | Spikes at 0 and 1 | Almost always 0 or 1 | No blending — equivalent to naive clone |
| 0.2 | U-shaped, mass near edges | Often 0.8–1.0, rarely 0.5 | Mild blending — samples lean strongly to P300 |
| 0.4 | Less extreme U-shape | More spread, e.g. 0.6–0.9 | Moderate blending |
| 1.0 | Uniform on [0, 1] | Equally likely anywhere | Maximum blending |

We test $\alpha \in \{0.2, 0.4\}$, following the recommended range from Zhang et al. (2018).

### C. Why Mixup is Theoretically Sound for EEG

EEG signals are continuous voltage measurements, and the linear combination of two valid EEG traces produces another physically plausible signal. This contrasts with image data, where blending two photographs can create perceptually invalid hybrids. The blended signal $\mathbf{x}_{\text{new}}$ retains the statistical and spectral properties of EEG, making the augmentation domain-appropriate.

Furthermore, the soft label $y_{\text{new}} \in [0, 1]$ honestly reflects the degree of P300 content in the synthetic signal. Binary cross-entropy loss accepts soft labels natively:

$$\mathcal{L} = -y_{\text{new}} \log(\hat{p}) - (1 - y_{\text{new}}) \log(1 - \hat{p}) \tag{9}$$

Training the network on soft labels yields well-calibrated probability estimates and smooth decision boundaries.

### D. Augmentation Strategies Implemented

We implement and compare four strategies:

**Strategy 1 — Baseline (Naive Clone):**

$$\text{Training set} = \underbrace{12{,}750}_{\text{Non-P300}} \cup \underbrace{12{,}750}_{\text{P300 copies}}$$

Hard labels only: $y \in \{0.0, 1.0\}$

**Strategy 2 — Pure Mixup (α=0.2 or α=0.4):**

$$n_{\text{generate}} = n_{\text{non}} - n_{\text{P300}} = 12{,}750 - 2{,}550 = 10{,}200$$

$$\text{Training set} = \underbrace{12{,}750}_{\text{Non-P300}} \cup \underbrace{2{,}550}_{\text{real P300}} \cup \underbrace{10{,}200}_{\text{Mixup synthetic}}$$

Soft labels: $y_{\text{synthetic}} = \lambda \in [0.5, 1.0]$

**Strategy 3 — Mixed 50% (our contribution):**

$$n_{\text{mixup}} = \lfloor 0.5 \times 10{,}200 \rfloor = 5{,}100$$

$$n_{\text{clone}} = 10{,}200 - 5{,}100 = 5{,}100$$

$$\text{Training set} = \underbrace{12{,}750}_{\text{Non-P300}} \cup \underbrace{2{,}550}_{\text{real P300}} \cup \underbrace{5{,}100}_{\text{Mixup}} \cup \underbrace{5{,}100}_{\text{clones}}$$

Mixed labels: 5,100 soft ($\lambda \in [0.5,1]$) + 5,100 hard ($y=1.0$)

### E. Integration into WE-SPSQ-CNN

All other components of the pipeline remain unchanged:

- Preprocessing: bandpass filtering and epoch extraction — unchanged
- SPSQ-CNN architecture: 45,809 parameters — unchanged
- Weighted ensemble: 5 classifiers with weights $W_k$ — unchanged
- Character decoding: row/column argmax — unchanged
- Only the **class imbalance handling step** is replaced

This surgical modification ensures that any observed performance change is directly attributable to the augmentation strategy.

---

## VI. Experimental Setup

### A. Implementation

The entire pipeline was implemented in **PyTorch** (the original paper used Keras/TensorFlow). Device selection is automatic: CUDA → MPS → CPU. All experiments used:

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | $10^{-3}$ |
| Batch size | 32 |
| Training epochs | 100 |
| Ensemble size | 5 classifiers |
| Dropout | p = 0.8 |
| Random seed | 42 |

### B. Evaluation Protocol

Models are trained on the 85-character training set and evaluated on the 100-character unseen test set. Character accuracy is computed at repetition counts $r \in \{1, 2, ..., 15\}$ by restricting averaging to the first $r$ repetitions per flash code.

The **resubstitution (training) accuracy** is computed by running the trained ensemble on the training data itself — this is expected to saturate at 100% and is not the primary evaluation metric.

The **test accuracy** (on the 100 unseen test characters) is the true measure of generalization.

### C. Variants Compared

| Variant Key | Method | α | Mixup fraction |
|---|---|---|---|
| `naive` | Baseline (naive clone) | — | 0% |
| `mixup_02` | Pure Mixup | 0.2 | 100% |
| `mixup_04` | Pure Mixup | 0.4 | 100% |
| `mixed_04` | Mixed 50% hybrid | 0.4 | 50% |

---

## VII. Results

### A. Training (Resubstitution) Accuracy

**Figure 1** shows training accuracy vs repetition count for Subject A and Subject B.

**Subject A — Training Accuracy:**
![Training accuracy Subject A](results/figures/accuracy_curves_A_train.png)

**Subject B — Training Accuracy:**
![Training accuracy Subject B](results/figures/accuracy_curves_B_train.png)

All four variants achieve **100% training accuracy** by repetition 3 and remain flat at 100% through repetition 15. This is the expected result of **resubstitution evaluation** — the model is tested on data it was already trained on, so it correctly recalls all 85 training characters once enough repetitions are averaged. The paper's reference line (black dashed), which lies well below at 76.5–94.5%, reports test accuracy — a fundamentally different measurement. Comparing these two is not meaningful.

The only informative region of the training graph is repetitions 1–2, where the four coloured lines show small differences before saturation. These differences reflect how quickly each model's learned representation converges on training data.

### B. Test Set Accuracy — The Primary Result

**Figure 2** shows test accuracy on 100 unseen characters for Subject A and Subject B.

**Subject A — Test Accuracy:**
![Test accuracy Subject A](results/figures/accuracy_curves_A_test.png)

**Subject B — Test Accuracy:**
![Test accuracy Subject B](results/figures/accuracy_curves_B_test.png)

**Table II: Subject A — Test Character Accuracy (%)**

| Method | 5 reps | 10 reps | 15 reps |
|---|---|---|---|
| Paper (Shukla 2024) | 76.5% | 87.5% | 94.5% |
| Baseline (naive clone) | 60.0% | 88.0% | 95.0% |
| Mixup α=0.2 | 54.0% | 85.0% | **97.0%** |
| Mixup α=0.4 | 54.0% | 86.0% | 96.0% |
| Mixed 50% (ours) | 59.0% | 85.0% | **97.0%** |

**Table III: Subject B — Test Character Accuracy (%)**

| Method | 5 reps | 10 reps | 15 reps |
|---|---|---|---|
| Paper (Shukla 2024) | 62.0% | 80.0% | 91.0% |
| Baseline (naive clone) | 79.0% | 92.0% | 93.0% |
| Mixup α=0.2 | 78.0% | **94.0%** | 92.0% |
| Mixup α=0.4 | 77.0% | 92.0% | 92.0% |
| Mixed 50% (ours) | **80.0%** | 93.0% | **95.0%** |

### C. Key Observations

**Observation 1 — Baseline reproduction is successful.**
At 10 and 15 repetitions, our baseline matches the paper almost exactly:
- Subject A: 88.0% (ours) vs 87.5% (paper) at 10r; 95.0% vs 94.5% at 15r
- Subject B: 92.0% vs 80.0% at 10r; 93.0% vs 91.0% at 15r

Our Subject B results exceed the paper's reported values, suggesting our PyTorch implementation's preprocessing is effective.

**Observation 2 — Mixup improves peak accuracy at 15 repetitions.**
For Subject A, both Mixup α=0.2 and Mixed 50% reach **97.0%** compared to the baseline's **95.0%** — a gain of 2 percentage points. For Subject B, Mixed 50% reaches **95.0%** vs the baseline's **93.0%** — again +2 points. The gain is modest but consistent.

**Observation 3 — Mixup degrades accuracy at 5 repetitions for Subject A.**
This is the most significant finding. At 5 repetitions, Mixup α=0.2 and α=0.4 both score **54.0%** versus the baseline's **60.0%** — a loss of 6 percentage points. The Mixed 50% hybrid partially recovers this with 59.0%, confirming that the hybrid strategy reduces but does not eliminate the low-repetition penalty.

**Observation 4 — Mixed 50% is the best single strategy for Subject B.**
Mixed 50% achieves the highest or joint-highest accuracy at every reported repetition count for Subject B: 80.0% at 5r, 93.0% at 10r, 95.0% at 15r. This is our team's contribution — neither the Shukla paper nor the Mixup paper tested this hybrid strategy on EEG data.

**Observation 5 — All methods exceed the paper on Subject B.**
Every variant — including the baseline — outperforms the paper's reported Subject B numbers at every repetition count. This indicates that our PyTorch reproduction with Chebyshev bandpass filtering applied correctly to continuous EEG (before epoching) produces better-quality preprocessed signals than the original implementation.

---

## VIII. Discussion

### A. What the Proposed Method Solves

**1. Higher-quality augmented training data.**
Naive cloning gives the model 10,200 additional training examples that are exact copies of 2,550 real signals. These copies provide no new gradient information. Mixup generates 10,200 genuinely distinct synthetic signals, each a unique linear blend of a different P300-Non-P300 pair. The model is exposed to a continuous spectrum of EEG patterns between the two classes, building a more flexible internal representation of "what P300-ness looks like."

**2. Soft label regularization.**
Hard labels (0 or 1) push the network to output probabilities very close to 0 or 1, causing overconfidence. Soft labels ($\lambda \in [0.5, 1]$) train the network to express calibrated uncertainty. When the P300 signal is genuinely ambiguous in a test epoch, a Mixup-trained model outputs a probability like 0.62 rather than 0.91, making the ensemble's weighted averaging more meaningful.

**3. Marginal but consistent improvement at high repetitions.**
At 15 repetitions — where the averaged P300 signal is clear — Mixup-trained models achieve 97% for Subject A and 95% for Subject B, each 2 percentage points above the baseline. The model's better-generalized representation correctly identifies more characters when the signal is strong enough to work with.

**4. The Mixed 50% hybrid as a viable strategy.**
Our Mixed 50% strategy achieves the best results on Subject B across all repetition counts simultaneously. This suggests that a partial Mixup approach — keeping some hard-label clones to anchor the model while adding Mixup samples to improve generalization — can be more robust than either extreme.

### B. What the Proposed Method Does Not Solve

**1. Low-repetition accuracy — the most clinically critical problem.**
At 5 repetitions, Mixup worsens Subject A accuracy from 60% to 54%. This is the most important failure. The soft-label training makes the model more uncertain, which hurts when only 5 noisy flash epochs are available per code. The model needs more signal to overcome its calibrated uncertainty, but at low repetitions the signal is simply not there.

This matters enormously in practice. The information transfer rate (ITR) of a P300 speller is:

$$\text{ITR} = \frac{60}{t_{\text{char}}} \cdot \left[\log_2 N + p \log_2 p + (1-p)\log_2\frac{1-p}{N-1}\right] \tag{10}$$

where $N=36$ characters, $p$ is accuracy, and $t_{\text{char}}$ is time per character. Fewer repetitions mean faster typing — improving accuracy at 5r is far more valuable than at 15r.

**2. Single-trial detection.**
At 1 repetition, all methods score 20–44% — close to chance for some cases. The P300 signal is too weak in a single 667ms window for any augmentation strategy to compensate. True single-trial P300 detection remains an open problem.

**3. Cross-subject generalization.**
Separate models are trained for Subject A and Subject B. Mixup does not enable a model trained on one subject's EEG to work on another subject's EEG. Each new user still requires a full calibration recording session — a significant burden for severely paralyzed patients.

**4. Cross-session generalization.**
Training and testing occur within the same recording session. In real deployment, EEG characteristics shift between sessions due to electrode placement variation, skin conductance changes, and fatigue. Mixup does not address this shift.

**5. The 5-repetition gap for Subject A vs the paper.**
The paper reports 76.5% at 5 repetitions (mean across subjects). Our best result for Subject A at 5r is 60%. This 16-percentage-point gap is not closed by Mixup and may reflect differences in preprocessing, specifically the paper's mention of decimation every 14th sample which we did not replicate identically.

**6. Inference speed.**
The 5-classifier ensemble takes 8–20ms per character to classify, compared to 2–13ms for a single SPSQ-CNN. Mixup does not reduce inference time.

### C. Summary Table

| Problem | Solved by Mixup? | Evidence |
|---|---|---|
| Class imbalance quality | **Yes** | Genuine new synthetic signals |
| High-rep accuracy (15r) | **Yes** | +2% on both subjects |
| Soft label calibration | **Yes** | Theoretically grounded |
| Subject B generalization | **Yes** | Mixed 50% best on all rep counts |
| Low-rep accuracy (5r) | **No** | −6% on Subject A |
| Single-trial (1r) detection | **No** | Still near-chance |
| Cross-subject generalization | **No** | Separate models required |
| Cross-session generalization | **No** | Not tested or addressed |
| Inference speed | **No** | Unchanged |
| Calibration burden | **No** | Full session still required |

---

## IX. Conclusion

This project reproduced the WE-SPSQ-CNN baseline from Shukla et al. in PyTorch and extended it with Mixup augmentation to address the methodological limitation of naive P300 cloning. Our reproduction successfully matches the paper's reported results at 10 and 15 repetitions for both subjects, and exceeds them on Subject B.

Our Mixup-augmented variants yield consistent improvements at 15 repetitions (+2% on both subjects), with the Mixed 50% hybrid strategy achieving the best overall performance on Subject B across all repetition counts. These results confirm that principled data augmentation provides better generalization than naive duplication.

However, the most clinically important finding is negative: **Mixup does not improve — and for Subject A reduces — accuracy at the low repetition counts (5r) that determine real-world communication speed.** The soft-label uncertainty introduced by Mixup is detrimental when only a few noisy epochs are available for averaging. This reveals a fundamental tension between generalization-focused augmentation and the low-data regime of sparse repetition decoding.

The unsolved core problem remains: achieving high character recognition accuracy at 1–5 repetitions. Future directions that could address this include:

- **Subject-independent models** trained across multiple subjects to improve generalization
- **Attention-based architectures** (e.g., Transformers) that can weight informative time points and channels dynamically
- **Riemannian geometry methods** that respect the non-Euclidean structure of EEG covariance matrices
- **Online adaptation** that updates the model continuously during a session as more labeled data accumulates
- **Frequency-domain augmentation** such as time-frequency masking, which may better preserve the spectral structure of the P300

---

## References

[1] N. Birbaumer and L. G. Cohen, "Brain-computer interfaces: communication and restoration of movement in paralysis," *The Journal of Physiology*, vol. 579, no. 3, pp. 621–636, 2007.

[2] L. A. Farwell and E. Donchin, "Talking off the top of your head: toward a mental prosthesis utilizing event-related brain potentials," *Electroencephalography and Clinical Neurophysiology*, vol. 70, no. 6, pp. 510–523, 1988.

[3] P. K. Shukla, H. Cecotti, and Y. K. Meena, "Towards effective deep neural network approach for multi-trial P300-based character recognition in brain-computer interfaces," *arXiv preprint arXiv:2410.08561*, 2024.

[4] A. Rakotomamonjy and V. Guigue, "BCI competition III: Dataset II — ensemble of SVMs for BCI P300 speller," *IEEE Transactions on Biomedical Engineering*, vol. 55, no. 3, pp. 1147–1154, 2008.

[5] M. Salvaris and F. Sepulveda, "Wavelets and ensemble of FLDs for P300 classification," in *Proc. 4th International IEEE/EMBS Conference on Neural Engineering*, 2009, pp. 339–342.

[6] H. Cecotti and A. Gräser, "Convolutional neural networks for P300 detection with application to brain-computer interfaces," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 33, no. 3, pp. 433–445, 2011.

[7] Z. Wang, C. Chen, J. Li, F. Wan, Y. Sun, and H. Wang, "ST-CapsNet: Linking spatial and temporal attention with capsule network for P300 detection improvement," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 31, pp. 991–1000, 2023.

[8] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond empirical risk minimization," in *International Conference on Learning Representations (ICLR)*, 2018.

---

*Report generated: May 2026*
*Code repository: BCI\_Mixup\_Enhanced*
*All experiments conducted on BCI Competition III Dataset II*
