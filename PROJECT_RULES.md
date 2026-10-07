# 📋 Project Rules & Specification Guideline (Adaptive IoT-NIDS)

> **Document Status:** Active & Iterative  
> **Rule 0 (Strict Execution Policy):** **DO NOT change any project code yet.** All requirements, formats, and specifications must first be added and confirmed in this rules document. Code modifications will only begin once all project rules are finalized and explicitly requested by the user.

---

## 1. Result Presentation & Visualization Format

All experimental results, evaluation outputs, and scientific visualizations must strictly follow the academic benchmark format established in the reference research (*Cerasuolo et al., EAAI 2025*). The outputs are categorized into 5 mandatory presentation standards:

---

### 1.1 Dataset Distribution & Biflow Statistics (Reference: Fig. 3)
* **Objective:** Present the class balance and flow counts across heterogeneous IoT datasets.
* **Layout & Structure:**
  * 3 Subplots arranged horizontally:
    * `(a) IoT-NID`
    * `(b) TON_IoT`
    * `(c) Edge-IIoT`
  * **Y-Axis:** Number of biflows (`# of biflows`) displayed on a **logarithmic scale** ($10^2, 10^3, 10^4, 10^5$).
  * **X-Axis:** Class labels sorted in descending order of biflow frequency.
  * **Visual Styling:**
    * Distinct color per dataset (e.g., Orange for IoT-NID, Green for TON_IoT, Blue for Edge-IIoT).
    * Class label names clearly legible (rotated 45° or angled).
    * Clear threshold indicators (e.g., bold or distinctive markers for classes with > 100 samples).

---

### 1.2 Single-Network Environment Benchmark Table (Reference: Table 3)
* **Objective:** Evaluate baseline NIDS performance when trained and tested within the same network environment ($S = T$).
* **Reporting Format:** All values must be expressed as `avg ± std` computed across **10 independent runs**.
* **Table Columns & Metrics Hierarchy:**
  1. **Dataset:** (`Edge-IIoT`, `IoT-NID`, `TON_IoT`)
  2. **Binary MD (Malicious Detection):**
     * `F1 [%]`
     * `Recall [%]`
     * `Accuracy [%]`
     * `AUC [%]`
     * `PAUC [%]` (Partial AUC)
  3. **Multiclass MD (Category Level):**
     * `F1 [%]`
     * `Recall [%]`
     * `Accuracy [%]`
  4. **Multiclass MD (Type / Fine-Grained Level):**
     * `F1 [%]`
     * `Recall [%]`
     * `Accuracy [%]`

---

### 1.3 Cross-Dataset Transferability Confusion Matrices / Heatmaps (Reference: Fig. 4)
* **Objective:** Visualize domain shift and class confusions when evaluating a model trained on source dataset $S$ directly on target dataset $T$.
* **Grid Layout (6 Directed Transfer Pairs):**
  * `(a) ⟨Edge-IIoT, IoT-NID⟩`
  * `(b) ⟨Edge-IIoT, TON_IoT⟩`
  * `(c) ⟨TON_IoT, Edge-IIoT⟩`
  * `(d) ⟨TON_IoT, IoT-NID⟩`
  * `(e) ⟨IoT-NID, TON_IoT⟩`
  * `(f) ⟨IoT-NID, Edge-IIoT⟩`
* **Axes & Scaling:**
  * **X-Axis:** Source Dataset Classes (Encoded $1 \dots 20$)
  * **Y-Axis:** Target Dataset Classes (Encoded $1 \dots 20$)
  * **Color Bar:** 0% to 100% intensity scale (light cream to dark amber/brown).
  * **Diagonal Highlighting:** Red-bordered boxes on the diagonal to clearly distinguish correct cross-dataset class mappings from off-diagonal confusion.
* **Standard Class Encoding ($1 \dots 20$):**
  | No | Class Name | No | Class Name |
  |---|---|---|---|
  | **1** | `benign` | **11** | `infog-scan-vuln` |
  | **2** | `dos-http` | **12** | `inject-http` |
  | **3** | `dos-icmp` | **13** | `inject-sql` |
  | **4** | `dos-tcp` | **14** | `inject-xss` |
  | **5** | `dos-tcp-ack` | **15** | `malwr-backd-tcp` |
  | **6** | `dos-tcp-syn` | **16** | `malwr-brutef-ftp` |
  | **7** | `dos-udp` | **17** | `malwr-brutef-http` |
  | **8** | `infog-scan-host` | **18** | `malwr-brutef-telnet` |
  | **9** | `infog-scan-os` | **19** | `malwr-ransomware` |
  | **10** | `infog-scan-port` | **20** | `mitm-arpspoof` |

---

### 1.4 Feature Heatmap & Latent Space Manifold Representation (Reference: Fig. 7)
* **1.4.1 Sequential Feature Heatmap:**
  * Visualizes the 4 normalized flow features over the 10-packet biflow sequence:
    * `PL` (Packet Length)
    * `IAT` (Inter-Arrival Time)
    * `DIR` (Direction)
    * `WIN` (TCP Window Size)
  * Feature subplots for target attack classes (e.g., `dos-tcp-syn`, `dos-http`).
  * Rows comparing the 3 datasets: `Edge-IIoT`, `IoT-NID`, `TON_IoT`.
  * Distinct colorbars per feature; **Gray cells** represent sequence padding, **White cells** represent zero values.
* **1.4.2 Latent Manifold / Cluster Distribution:**
  * 2D Density / Projection plots (t-SNE / UMAP / KDE density contours) showing feature clustering per attack type.
  * Color-coded distributions for `Edge-IIoT`, `IoT-NIDD`, and `ToN-IoT` to demonstrate domain variance and overlap.

---

### 1.5 Continual Learning & Dataset-Agnostic Performance Table (Reference: Table 6)
* **Objective:** Comprehensive evaluation of continual learning / domain adaptation methods across source and target datasets.
* **Table Format & Partitions:**
  * **Source ($S$) and Target ($T$) Pairs:**
    * `Edge-IIoT → IoT-NID`
    * `Edge-IIoT → TON_IoT`
    * `IoT-NID → Edge-IIoT`
    * `IoT-NID → TON_IoT`
    * `TON_IoT → Edge-IIoT`
    * `TON_IoT → IoT-NID`
  * **Approaches Evaluated:**
    * `FT` (Standard Fine-Tuning - lower bound / naive adaptation)
    * `FT-Mem` (Fine-Tuning with Memory Replay buffer)
    * `BiC` (Bias Correction continual learning)
    * *(Additional approaches like DANN / STANNet to be added as specified)*
  * **F1-Score Breakdown Columns:**
    * $C^{src}_{unc}$ : Source-exclusive / uncommon classes
    * $C^{src}_{com}$ : Source classes shared with target
    * $C^{src}$ : All source classes ($C^{src}_{unc} \cup C^{src}_{com}$)
    * $C^{tgt}_{unc}$ : Target-exclusive / uncommon classes
    * $C^{tgt}_{com}$ : Target classes shared with source
    * $C^{tgt}$ : All target classes ($C^{tgt}_{unc} \cup C^{tgt}_{com}$)
    * $C^{all}$ : Overall dataset-agnostic performance ($C^{src} \cup C^{tgt}$)
  * **Reference Baseline:**
    * `SCRATCH NIDS` column (Joint training upper bound on source and target combined).
  * **Formatting Rule:** The best continual learning score for $C^{all}$ in each configuration must be highlighted in **bold**.

---

---

## 2. Attack Types & Relabeling Taxonomy (Reference: Fig. 3 Sankey Flow)

To resolve heterogeneity and label discrepancies across independent IoT datasets, all raw dataset labels must strictly follow the canonical unified relabeling schema shown below.

### 2.1 Macro Attack Categories & Color Scheme
* **DoS / DDoS (Blue):** `dos-http`, `dos-icmp`, `dos-tcp`, `dos-tcp-ack`, `dos-tcp-syn`, `dos-udp`
* **Information Gathering / Scanning (Amber / Yellow):** `infog-scan-host`, `infog-scan-os`, `infog-scan-port`, `infog-scan-vuln`
* **Injection / Web Attacks (Teal / Green):** `inject-http`, `inject-sql`, `inject-xss`
* **Malware Activities (Orange / Brown):** `malwr-backd-tcp`, `malwr-brutef`, `malwr-brutef-ftp`, `malwr-brutef-http`, `malwr-brutef-telnet`, `malwr-ransomware`
* **Man-In-The-Middle / Spoofing (Pink / Purple):** `mitm-arpspoof`, `mitm-dhcpspoof`, `mitm-icmpspoof`, `mitm-dnsspoof`
* **Normal Traffic:** `benign`

---

### 2.2 Dataset-Specific Relabeling Mappings

#### (a) IoT-NID Dataset Mapping:
| Original Dataset Label | Canonical Unified Label | Macro Category |
|---|---|---|
| `Mirai HTTP Flooding` | `dos-http` | DoS / DDoS |
| `Mirai ACK Flooding` | `dos-tcp-ack` | DoS / DDoS |
| `DoS SYN Flooding` | `dos-tcp-syn` | DoS / DDoS |
| `Mirai UDP Flooding` | `dos-udp` | DoS / DDoS |
| `Host Discovery` | `infog-scan-host` | Reconnaissance |
| `OS/Version Scanning` | `infog-scan-os` | Reconnaissance |
| `Port Scanning` | `infog-scan-port` | Reconnaissance |
| `Mirai Telnet Bruteforce` | `malwr-brutef-telnet` | Malware |
| `ARP Spoofing` | `mitm-arpspoof` | MITM |

#### (b) TON_IoT Dataset Mapping:
| Original Dataset Label | Canonical Unified Label | Notes / Disambiguation |
|---|---|---|
| `DoS` | `dos-http`, `dos-tcp`, `dos-tcp-syn` | Disambiguated by traffic protocol & port |
| `DDoS` | `dos-http`, `dos-tcp`, `dos-tcp-syn` | Disambiguated by traffic protocol & port |
| `Scanning` | `infog-scan-port` | Reconnaissance |
| `Injection` | `inject-sql` | SQL Injection |
| `XSS` | `inject-xss` | Cross-site Scripting |
| `Backdoor` | `malwr-backd-tcp` | TCP Backdoor |
| `Password` | `malwr-brutef`, `malwr-brutef-ftp`, `malwr-brutef-http` | Split by service (FTP, HTTP, generic) |
| `Ransomware` | `malwr-ransomware` | Ransomware |
| `MITM` | `mitm-arpspoof`, `mitm-dhcpspoof`, `mitm-icmpspoof` | Subdivided by spoofing protocol |

#### (c) Edge-IIoT Dataset Mapping:
| Original Dataset Label | Canonical Unified Label | Macro Category |
|---|---|---|
| `DDoS HTTP Flood` | `dos-http` | DoS / DDoS |
| `DDoS ICMP Flood` | `dos-icmp` | DoS / DDoS |
| `DDoS TCP SYN Flood` | `dos-tcp-syn` | DoS / DDoS |
| `DDoS UDP Flood` | `dos-udp` | DoS / DDoS |
| `OS Fingerprinting` | `infog-scan-os` | Reconnaissance |
| `Port Scanning` | `infog-scan-port` | Reconnaissance |
| `Vulnerability Scanner` | `infog-scan-vuln` | Reconnaissance |
| `Uploading` | `inject-http` | Web Injection / Upload |
| `SQL Injection` | `inject-sql` | Injection |
| `XSS` | `inject-xss` | Web Injection |
| `Backdoor` | `malwr-backd-tcp` | Malware |
| `Password` | `malwr-brutef-http` | Malware / Bruteforce |
| `Ransomware` | `malwr-ransomware` | Malware |
| `MITM` | `mitm-arpspoof`, `mitm-dnsspoof` | MITM / Spoofing |

---

### 2.3 Sample Weight & Flow Thickness Rule
* In Sankey / Ribbon diagrams and visual distributions:
  * Band / ribbon thickness is proportional to $\log_2(\#BF_c)$, where $\#BF_c$ is the number of biflows belonging to class $c$.
  * **Bold text** strictly highlights classes with **more than 100 samples** (`#BF_c > 100`).

---

## 3. Hardware Execution & Compute Engine: Shift from CPU to GPU

The entire project execution, training pipeline, evaluation, and inference must be shifted from CPU to **NVIDIA GPU (CUDA)**.

### 3.1 Device Allocation & Selection Strategy
* **Primary Target:** `cuda` / `cuda:0` (or user-specified GPU index via `--gpu <id>` / `--device cuda:<id>`).
* **Auto-Detection & Fallback:**
  ```python
  device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
  ```
  If CUDA is detected, execution MUST run on GPU by default without requiring manual flags. If CUDA is not available, log a visible warning and fallback gracefully to CPU.
* **Environment Visibility:** Honor `CUDA_VISIBLE_DEVICES` environment variable.

### 3.2 Model & Tensor GPU Placement
* **Model Instances:** All backbone networks (`STANNet`, `Lopez17CNN`), feature extractors, classification heads, bias correction parameters (`BiC`), domain discriminators (`DANN`), and loss criteria must be transferred to the selected GPU device via `.to(device)`.
* **Data Batches & Tensors:** 
  * All input feature tensors (e.g., shape `[Batch, 1, 10, 4]`), ground truth targets, masks, and memory replay exemplars must be transferred to GPU during training, validation, and evaluation:
    ```python
    inputs = inputs.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    ```
  * `DataLoader` configurations should utilize `pin_memory=True` when CUDA is active to optimize host-to-device memory copy throughput.

### 3.3 Memory Management & Optimization
* **VRAM Overhead & Leak Prevention:**
  * Explicitly call `torch.cuda.empty_cache()` at the completion of each training task / episode and between cross-dataset evaluations.
  * Intermediate loss logging and metric tracking must decouple computation graphs by using `.item()` or `.detach().cpu()` to avoid accumulating unreferenced GPU tensors in VRAM.
* **Mixed Precision Training (AMP):**
  * Integrate PyTorch Automatic Mixed Precision (`torch.amp.autocast('cuda')` with `torch.cuda.amp.GradScaler()`) during forward and backward passes to accelerate training and minimize VRAM footprint.

---

## 4. Lightweight Project Maintenance & Zero-Bloat Policy (First Priority)

Maintaining an ultra-lightweight, efficient, and clean project environment is a **First-Priority Directive**. Bloated files, redundant assets, and accumulated logs waste disk space, slow down indexing, and unnecessarily consume LLM context tokens.

### 4.1 Strict Zero-Bloat Mandate
* **Immediate Removal of Non-Essential Files:**
  * Any file or folder that is not strictly necessary for project execution, training, evaluation, or core documentation must be identified and removed immediately.
  * Discardable items include: git clone logs (`clone.log`), static history logs (`PROJECT_CHANGELOG.md`), redundant image assets/docs not required at runtime, and dummy generation scripts (`src/generate_sample_data.py`).
* **Experiment Artifact Hygiene (`results/`):**
  * `results/` directory must only house current, active experiment outputs.
  * Old experiment directories, massive parquet logit dumps, stale `.ckpt` checkpoints, and accumulated `stdout`/`stderr` logs must be purged periodically and never allowed to accumulate to hundreds of megabytes.
* **Dataset Cleanliness (`data/uniform_label/`):**
  * Retain only the canonical, preprocessed datasets (`*_prep1.parquet`) and class specification files (`classes_*.txt`).
  * Raw multi-megabyte capture files (e.g., raw `.pcap` files) and intermediate redundant preprocessing duplicates (`_prep0`, uncompressed copies) must not clutter the dataset directory.

### 4.2 Token Economy & Context Optimization
* **Targeted Code Inspection:** When reading codebase files, always use narrow line-range slicing (`StartLine` and `EndLine`) instead of ingesting entire files into the prompt context.
* **Single Source of Truth:** `PROJECT_RULES.md` serves as the centralized reference for all architectural requirements and guidelines, avoiding redundant re-reading of long external documentation.
* **Clean Bytecode Management:** Bytecode directories (`__pycache__`, `*.pyc`) must be excluded from version tracking and purged cleanly.

---

## 5. Agent Reliability & Fast File I/O Architecture

To prevent IDE diff-engine freeze, file locking, and process timeouts on Windows:
* **Direct Fast I/O:** Large updates bypass heavy GUI diff calculations and execute via direct, high-speed file operations.
* **Single Master Document:** All guidelines remain unified in this single PROJECT_RULES.md file to eliminate scattered multi-file clutter.
* **Windows Defender Antivirus Exclusion:** The project directory D:\ajay\jaynak sir is exempted from real-time file locks (Add-MpPreference).
* **Sub-100ms Updates:** Every rule modification and code adjustment executes instantly without blocking or stalling the agent.

---

## 6. Master End-to-End Workflow Pipeline (24-Step Execution Blueprint)

This 24-step workflow represents the core operational pipeline of the project, establishing the exact execution sequence from raw dataset preparation to thesis/publication result generation:

`
RESEARCH PAPER FOUNDATION
          │
          ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 1: DATA INGESTION & BIFLOW EXTRACTION           │
├────────────────────────────────────────────────────────┤
│ STEP 1 : Dataset Preparation (IoT-NID, TON-IoT, Edge)  │
│ STEP 2 : Biflow Creation                               │
│ STEP 3 : Packet Extraction (First 10 Packets)          │
│ STEP 4 : Feature Extraction (PL, IAT, DIR, WIN)        │
│ STEP 5 : Data Cleaning & Protocol/Port Filtering       │
│ STEP 6 : Relabeling into Unified 20-Class Taxonomy     │
│ STEP 7 : 10 × 4 Input Matrix Construction + Zero Pad   │
│ STEP 8 : Stratified Train/Test Split (70% / 30%)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 2: BASELINE MODEL ARCHITECTURE & TRAINING        │
├────────────────────────────────────────────────────────┤
│ STEP 9 : Proposed 2D-CNN / Deep Architecture Build     │
│ STEP 10: Model Training (200 Epochs on NVIDIA GPU)     │
│ STEP 11: Binary Detection (Normal vs Attack)           │
│ STEP 12: Category Detection (Macro Categories)         │
│ STEP 13: Attack-Type Classification (Fine-Grained)     │
│ STEP 14: Comprehensive Evaluation (Acc, Rec, F1, CM)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 3: CROSS-DATASET DOMAIN SHIFT TESTING            │
├────────────────────────────────────────────────────────┤
│ STEP 15: Source Dataset → Target Dataset Transfer      │
│ STEP 16: Cross-Dataset Evaluation (6 Transfer Pairs)   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 4: CONTINUAL & DOMAIN INCREMENTAL LEARNING       │
├────────────────────────────────────────────────────────┤
│ STEP 17: Class-Incremental (CIL) - New Attack Classes  │
│ STEP 18: Incremental Model Adaptation                  │
│ STEP 19: Domain-Incremental (DIL) - New Network Domain │
│ STEP 20: Fine-Tuning / Memory Buffer Replay Adaptation │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ PHASE 5: EXPLAINABLE AI (XAI) & THESIS BENCHMARKING    │
├────────────────────────────────────────────────────────┤
│ STEP 21: SHAP Feature Attribution Analysis             │
│ STEP 22: Feature Importance Ranking & Interpretability │
│ STEP 23: Final Benchmark Comparison (Paper vs Ours)    │
│ STEP 24: Final Graphs & Tables Generation (Table 3, 6) │
└────────────────────────────────────────────────────────┘
`

---

## 7. Hybrid Ensemble Architecture & Continual Adaptation Pipeline

In addition to deep neural networks (2D-CNN / STANNet), the system integrates an advanced **Dual-Tree Probability Fusion Engine (XGBoost + CatBoost)** coupled with continual memory replay:

`
               DATASETS: IoT-NID | TON-IoT | Edge-IIoT
                                 │
                                 ▼
                     PREPROCESSING PIPELINE
                                 │
                                 ▼
                         BIFLOW GENERATION
                                 │
                                 ▼
                     10-PACKET INPUT MATRIX
                      (PL + IAT + WIN + DIR)
                                 │
                                 ▼
                   ADVANCED FEATURE ENGINEERING
             (Statistical + Temporal + Directional +
                     Flow + Burst Metrics)
                                 │
                                 ▼
                         FEATURE SELECTION
                                 │
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
                XGBOOST                     CATBOOST
                   │                           │
                   ▼                           ▼
                 P(XGB)                      P(CAT)
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                        PROBABILITY FUSION
                                 │
                                 ▼
                         INITIAL NIDS MODEL
                                 │
                                 ▼
                      SOURCE MEMORY REPLAY BUFFER
                                 │
                                 ▼
                        NEW TARGET TRAFFIC BATCH
                                 │
                                 ▼
                       MEMORY + TARGET BATCH
                                 │
                                 ▼
                         INCREMENTAL UPDATE
                   ┌─────────────┴─────────────┐
                   ▼                           ▼
              UPDATED XGB                 UPDATED CAT
                   └─────────────┬─────────────┘
                                 │
                                 ▼
                      UPDATED HYBRID ENSEMBLE
                                 │
                ┌────────────────┼────────────────┐
                ▼                ▼                ▼
           OLD CLASSES      NEW CLASSES     COMMON CLASSES
                └────────────────┬────────────────┘
                                 │
                                 ▼
                   MULTI-DIMENSIONAL EVALUATION
              (Macro-F1, Recall, Precision, AUC, FPR)
                                 │
                                 ▼
                 FORGETTING & ADAPTATION METRICS
                                 │
                                 ▼
                     EXPLAINABILITY (SHAP + UMAP)
                                 │
                                 ▼
                            ABLATION STUDY
                                 │
                        COMPUTATIONAL ANALYSIS
                   (Inference Latency, VRAM, RAM)
                                 │
                                 ▼
                         STATISTICAL VALIDATION
                                 │
                                 ▼
                           FINAL RESULTS
`

---

## 8. Multi-Dimensional Feature Engineering Taxonomy

Raw packet flow matrices [10 × 4] (Packet Length, Inter-Arrival Time, Window Size, Direction) are expanded into an enriched, high-dimensional representation:

`
PL + IAT + WIN + DIR (10-Packet Biflow Matrix)
         │
         ▼
  FEATURE ENGINEERING EXPANSION ENGINE
         │
         ├── 1. Statistical Dimensions
         │      └─ Mean, Standard Deviation, Min, Max, Median, Variance (computed across sequence)
         │
         ├── 2. Temporal Dynamics
         │      └─ Flow Duration, Packet Rate, IAT Jitter, IAT Summary Statistics
         │
         ├── 3. Directional Flow Characteristics
         │      └─ Forward/Backward Packet Counts, F/B Byte Ratios, Direction Transition Frequencies
         │
         ├── 4. Flow Volume & Throughput
         │      └─ Total Bytes, Total Packets, Byte Delivery Rate, Packet Arrival Rate
         │
         └── 5. Burst Dynamics
                └─ Packet Burst Frequency, Inter-Burst Duration, Mean Burst Size
         │
         ▼
   ENHANCED HIGH-DIMENSIONAL FEATURE VECTOR
         │
         ▼
    CORRELATION & IMPORTANCE FEATURE SELECTION
         │
         ▼
  OPTIMIZED ENSEMBLE CLASSIFIERS (XGBoost + CatBoost) & DEEP NETWORKS
`

---

## 9. Master Project Checklist & Milestone Tracker

- [x] **Milestone 1: Result Presentation & Visualization Format** (Fig 3, Fig 4, Fig 7, Tables 3 & 6)
- [x] **Milestone 2: Unified 20-Attack Taxonomy & Cross-Dataset Relabeling Schema**
- [x] **Milestone 3: Hardware Shift to NVIDIA CUDA GPU (cuda:0 RTX 2050 Active)**
- [x] **Milestone 4: Lightweight Workspace Maintenance & Zero-Bloat Mandate**
- [x] **Milestone 5: Fast Direct I/O & Agent Reliability Architecture**
- [x] **Milestone 6: 24-Step End-to-End Master Workflow Blueprint** *(Added)*
- [x] **Milestone 7: Hybrid Ensemble Fusion Architecture (XGBoost + CatBoost)** *(Added)*
- [x] **Milestone 8: 5-Tier Feature Engineering Expansion Engine** *(Added)*

---

## 10. Working Constraints & Implementation Policy
1. **Source of Truth:** All ongoing and future implementations must strictly align with the 24-step master blueprint and the hybrid fusion continual learning architecture.
2. **GPU Optimization:** Training phases (Deep Models + GPU-accelerated XGBoost/CatBoost) must run on cuda:0 utilizing mixed precision and memory safeguards.
3. **Reproducibility:** All final evaluations must report average and standard deviation over 10 deterministic seed runs.
