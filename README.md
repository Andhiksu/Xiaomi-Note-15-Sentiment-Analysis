# [YouTube](https://www.youtube.com) Review Sentiment Analysis for Enterprise Insights
![Python](https://img.shields.io/badge/Python-3.11-blue)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SetFit-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-VM-blue)
![Microsoft Azure](https://img.shields.io/badge/Azure-VM-0078D4)
![LLM](https://img.shields.io/badge/LLM-Gemini%20API-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End to end NLP pipeline untuk menganalisis persepsi pasar terhadap produk smartphone berdasarkan komentar review [YouTube](https://www.youtube.com).

Proyek ini mengubah ribuan opini pengguna menjadi **insight strategis yang dapat digunakan oleh tim produk, marketing, dan engineering untuk pengambilan keputusan berbasis data.**

Studi kasus: **[Xiaomi Note 15 Series](https://www.mi.co.id/id/event/redmi-note-15-series/?utm_campaign=micom_id_search_brand&utm_source=google&utm_medium=paid-search&utm_type=1&utm_content=rn15_0122-0228&utm_id=2325&gad_source=1&gad_campaignid=1766176000&gbraid=0AAAAAC9UrG1GT_rml6bMwfuTuLIHFRXW_&gclid=Cj0KCQiA-YvMBhDtARIsAHZuUzIfccDQHqSIW9QL3cQlgclHA2lBOH539bvOnOzGdk0gMC80ztQRPKgaApabEALw_wcB)**

**Catatan:**  
File artefak training dan dataset berukuran besar tidak disertakan dalam repository ini untuk menjaga ukuran repository tetap ringan dan mudah diakses. Untuk mengakses file versi lengkap (termasuk dataset dan artifacts), silakan unduh melalui Google Drive berikut: [Link Google Drive](https://drive.google.com/drive/folders/1E2ck24tSE-F20OZ7SpKXIl8G-yECTRD8?usp=sharing)

---

# Business Problem

Review teknologi di [YouTube](https://www.youtube.com) sering menghasilkan ribuan komentar pengguna yang berisi opini tentang produk.

Namun bagi perusahaan, tantangan utamanya adalah:

- Bagaimana memahami **persepsi pasar secara cepat dan objektif?**
- Fitur produk mana yang paling banyak **dipuji atau dikeluhkan?**
- Apa **prioritas perbaikan produk** yang paling berdampak?

Proyek ini dirancang untuk menjawab pertanyaan tersebut menggunakan pendekatan **Natural Language Processing dan Machine Learning.**

---

# Project Objectives

Project ini bertujuan membangun pipeline NLP end-to-end yang mampu mengubah komentar YouTube menjadi insight bisnis yang terstruktur dan actionable.

Fokus utamanya:
- **Diagnose customer perception at aspect level**, Mengidentifikasi persepsi pelanggan pada level fitur seperti desain, kamera, harga, performa, software, baterai, layar, thermal, dan charging.
- **Prioritize high-impact pain points**, Menentukan aspek yang paling mendesak berdasarkan volume percakapan, intensitas sentimen, dan posisi relatif antar fitur.
- **Extract positive competitive signals**, Menemukan fitur yang relatif lebih kuat untuk dijadikan bahan positioning atau communication strategy.
- **Improve model reliability efficiently**, Meningkatkan kualitas klasifikasi melalui kombinasi hybrid baseline, LLM label assist, manual correction, SetFit, dan active learning.
- **Translate analysis into strategic actions**, Mengubah output teknis menjadi executive summary dan rekomendasi bisnis yang mudah dipahami stakeholder.

---

# Key Results

- **4,722** komentar YouTube dianalisis
- **1,071** sampel labeling dibuat dengan distribusi seimbang (357 per kelas)
- **94.40%** agreement antara LLM assist dan human-corrected labels dengan durasi 2.14 menit menggunakan metode eksekusi Paralel dengan `ThreadPoolExecutor`
- Model improved from: Accuracy: **0.52** → **0.80** dan Macro F1: **0.48** → **0.68**
- Pain point utama teridentifikasi pada: **Desain** & **Kamera**
- Output akhir mencakup:
  - **Sentiment Classification**
  - **Aspect Level Insight**
  - **Z-Score Prioritization**
  - **Executive LLM Report**
  - **Streamlit Dashboard**
  
---
# Dataset Summary

| Sentiment     | Baseline count | Baseline (%) | Final count | Final (%) | Selisih (%) |
|:--------------|---------------:|-------------:|------------:|----------:|------------:|
| **Neutral**   |           2265 |        47.97 |        3219 |     68.17 |      **+20.20** |
| **Negative**  |           1351 |        28.61 |        1231 |     26.07 |       −2.54 |
| **Positive**  |           1106 |        23.42 |         272 |      5.76 |     **−17.66** |
| **Total**     |           4722 |       100.00 |        4722 |    100.00 |             —   |

> Distribusi final menunjukkan bahwa mayoritas komentar sebenarnya bersifat netral-informatif, sementara kelas positive menjadi jauh lebih kecil setelah model diperbaiki. Ini mengindikasikan bahwa model final lebih ketat dan lebih realistis dalam membedakan pujian, opini informatif, dan keluhan.
---

## Pipeline Overview

1.	**Data Collection**, Mengumpulkan komentar menggunakan [YouTube Data API v3](https://developers.google.com/youtube/v3/getting-started) dari beberapa channel reviewer gadget Indonesia.
2.	**Text Preprocessing**, Membersihkan teks melalui normalisasi slang, penghapusan emoji/karakter tidak relevan, dan filtering komentar kosong.
3.	**Hybrid Baseline Sentiment Analysis**, Menggunakan [IndoBERT](https://huggingface.co/w11wo/indonesian-roberta-base-sentiment-classifier) + keyword boosting sebagai baseline awal.
4.	**LLM-Assisted Labeling**, Membuat template labeling terstratifikasi sebanyak **1,071 sampel** dengan menggunakan [Gemini 3 Flash Preview](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/) untuk memberi pre-label dan catatan awal sebelum dikoreksi manusia.
5.	**Gold Dataset & Fixed Split**, Menyusun gold dataset hasil koreksi manusia lalu membaginya menjadi:
   - Train: **642**
	- Validation: **214**
	- Fixed Test: **215**
7.	**Fine-Tuning**, Menggunakan model [SetFit](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) agar lebih adaptif terhadap bahasa komentar YouTube Indonesia.
8.	**Active Learning**, Menambah data paling informatif melalui:
	- Uncertainty sampling
	- Positive mining
	- Disagreement cases
9.	**Aspect-Based Categorization**, Memetakan komentar ke aspek produk seperti `desain`, `kamera`, `harga`, `software`, `performa`, `baterai`, `layar`, `thermal`, dan `charging`.
10.	**Z-Score Prioritization**, Mengukur posisi relatif tiap aspek untuk menentukan pain point dan relative strengths secara lebih objektif.
11. **Executive Reporting & Dashboard**, Mengubah insight teknis menjadi narasi strategis dengan [Gemini API](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/) dan menampilkannya dalam dashboard [Streamlit](https://streamlit.io).

---

# Model Performance

Evaluasi dilakukan pada fixed test set (215 data) agar perbandingan model tetap fair.

### Ringkasan Performa Keseluruhan
| Model              | Accuracy | Macro F1 |
|--------------------|----------|----------|
| Baseline Hybrid    | 0.52     | 0.48     |
| SetFit Round 0     | **0.80** | **0.67** |
| SetFit Round 1     | **0.80** | **0.68** |

#### Per-Class F1-Score (Fixed Test Set)
| Model              | Negative | Neutral | Positive |
|--------------------|----------|---------|----------|
| Baseline Hybrid    | 0.58     | 0.58    | 0.29     |
| SetFit Round 0     | 0.71     | **0.89**| 0.41     |
| SetFit Round 1     | **0.75** | 0.88    | 0.41     |

>Fine-tuning memberi lompatan performa terbesar, sedangkan Active Learning membantu memperbaiki kualitas klasifikasi antar kelas, terutama pada area yang sebelumnya menjadi blind spot model.

---

# Aspect-Based Insight Summary
Berikut ringkasan aspek (diurutkan berdasarkan total mention descending):

| Aspek       | Total Mention | Avg Sentiment | % Negative | % Neutral | % Positive | Z-Score     | Status Z-Score | Status Absolut (berdasarkan avg) |
|:------------|---------------|---------------|------------|-----------|------------|-------------|----------------|----------------------------------|
| **desain**  | 519           | -0.809        | 84.0%      | 12.9%     | 3.1%       | **-1.95**   | 🔴 Negatif     | 🔴 Negatif                       |
| **kamera**  | 489           | -0.654        | 68.7%      | 28.0%     | 3.3%       | **-1.28**   | 🔴 Negatif     | 🔴 Negatif                       |
| **harga**   | 272           | -0.452        | 53.3%      | 38.6%     | 8.1%       | -0.41       | 🟡 Netral      | 🔴 Negatif                       |
| **performa**| 188           | -0.256        | 33.5%      | 58.5%     | 8.0%       | +0.44       | 🟡 Netral      | 🔴 Negatif                       |
| **software**| 163           | -0.270        | 30.1%      | 66.9%     | 3.1%       | +0.38       | 🟡 Netral      | 🔴 Negatif                       |
| **baterai** | 162           | -0.160        | 22.8%      | 70.4%     | 6.8%       | **+0.85**   | 🟢 Positif     | 🔴 Negatif                       |
| **layar**   | 161           | -0.217        | 34.2%      | 53.4%     | 12.4%      | +0.60       | 🟢 Positif     | 🔴 Negatif                       |
| **thermal** | 31            | -0.194        | 22.6%      | 74.2%     | 3.2%       | +0.71       | 🟢 Positif     | 🔴 Negatif                       |
| **charging**| 20            | -0.200        | 20.0%      | 80.0%     | 0.0%       | +0.68       | 🟢 Positif     | 🔴 Negatif                       |



---
# Training Infrastructure

Performa fine-tuning diuji pada beberapa environment cloud:

| No | Hardware                        | Spesifikasi                                                                 | Mode Training     | Waktu Training (1 run) | Keterangan                                                                 |
|----|---------------------------------|-----------------------------------------------------------------------------|-------------------|------------------------|----------------------------------------------------------------------------|
| 1  | Local – MacBook Air M3  | • CPU 8-core<br>• GPU hingga 10-core<br>• RAM 16 GB                        | CPU-based         | > 387 menit                | Cukup untuk eksperimen awal, namun proses fine-tuning relatif lambat karena keterbatasan resource |
| 2  | Microsoft Azure VM              | • 16 core CPU<br>• 32 thread<br>• 128 GB RAM                               | CPU-based         | ± 32 menit             | Peningkatan jumlah core dan RAM mempercepat proses secara signifikan dibanding local machine |
| 3  | Google Cloud VM (GPU-Enabled)   | • 2 core CPU<br>• 4 thread<br>• 16 GB RAM<br>• Nvidia L4 GPU              | GPU-based         | ± 17 menit             | GPU (Nvidia L4) memberikan percepatan ~2× dibanding CPU cloud dan jauh lebih efisien dibanding local machine |

Benchmark ini menunjukkan peningkatan efisiensi training secara signifikan ketika menggunakan GPU.

---
# Future Improvement

Arah pengembangan berikutnya:
- Integrasi enterprise-grade LLM untuk reporting yang lebih stabil
- Ekspansi ke platform lain seperti X, TikTok, Instagram
- Real-time monitoring untuk sentiment spike dan pain point tracking
- Penguatan aspect intelligence, termasuk competitor mention dan complaint theme clustering


---
# Run the Project

### **Installation**
1. Install Environment:
```text
conda env create -f environment.yml
conda activate app-env
```
2. Set YouTube & Gemini API Key:
```text
YOUTUBE_API_KEY=your_api_key_here
GEMINI_API_KEY=your_api_key_here
```
3. Run Dashboard (Streamlit):
```text
streamlit run dashboard.py
```

---
