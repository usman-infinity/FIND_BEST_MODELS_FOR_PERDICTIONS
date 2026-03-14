# Find Best Models for Predictions 📈

This repository provides a comprehensive framework for evaluating and benchmarking deep learning architectures for time-series forecasting. The project focuses on predicting environmental data (PM10 levels) by comparing state-of-the-art models.

## 🏆 The Winner: Informer-CNN-BiLSTM Hybrid
Through extensive testing, the **Informer-CNN-BiLSTM** hybrid architecture emerged as the top performer, achieving a **4x reduction in error** compared to standard Transformers.

### Key Features of the Winning Model:
* **CNN Layer:** Extracts local temporal features and filters input noise.
* **Informer Attention:** Uses ProbSparse self-attention for efficient long-range dependency modeling.
* **BiLSTM Layer:** Captures bidirectional context to refine final predictions.

---

## 📊 Benchmark Results
| Model Architecture | Val MAE | Remarks |
| :--- | :--- | :--- |
| **CNN-BiGRU** | 0.0236 | Baseline stability |
| **TCN** | 0.0228 | Strong long-range capture |
| **Enhanced Transformer**| 0.0244 | High capacity, slower convergence |
| **Informer-CNN-BiLSTM** | **0.0064** | **Optimal Performance** |

