# YouTube Review Sentiment Analysis for Enterprise Insights
![Python](https://img.shields.io/badge/Python-3.11-blue)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-SetFit-green)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-VM-blue)
![Microsoft Azure](https://img.shields.io/badge/Azure-VM-0078D4)
![LLM](https://img.shields.io/badge/LLM-Gemini%20API-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end NLP pipeline untuk menganalisis persepsi pasar terhadap produk smartphone berdasarkan komentar review YouTube.

Proyek ini mengubah ribuan opini pengguna menjadi **insight strategis yang dapat digunakan oleh tim produk, marketing, dan engineering untuk pengambilan keputusan berbasis data.**

Studi kasus: **Xiaomi Note 15 Series**

**Catatan:**  
File artefak training dan dataset berukuran besar tidak disertakan dalam repository ini untuk menjaga ukuran repository tetap ringan dan mudah diakses. Untuk mengakses versi lengkap file proyek (termasuk dataset dan artifacts), silakan unduh melalui Google Drive berikut: [Link Google Drive](https://drive.google.com/drive/folders/1Hg7Y_iZXviButfFkWlp_TIfZd7MrdJvM?usp=sharing)

---

# Business Problem

Review teknologi di YouTube sering menghasilkan ribuan komentar pengguna yang berisi opini tentang produk.

Namun bagi perusahaan, tantangan utamanya adalah:

- Bagaimana memahami **persepsi pasar secara cepat dan objektif?**
- Fitur produk mana yang paling banyak **dipuji atau dikeluhkan?**
- Apa **prioritas perbaikan produk** yang paling berdampak?

Proyek ini dirancang untuk menjawab pertanyaan tersebut menggunakan pendekatan **Natural Language Processing dan Machine Learning.**

---

# Why This Project Matters

Dalam industri teknologi, memahami persepsi pengguna sangat penting untuk:

• meningkatkan kualitas produk  
• memperbaiki pengalaman pengguna  
• menentukan strategi pemasaran  

Proyek ini menunjukkan bagaimana **machine learning dan NLP dapat digunakan untuk mengekstrak insight strategis dari ribuan opini pengguna secara otomatis.**

Pendekatan ini dapat diterapkan untuk berbagai kebutuhan seperti:

- Product feedback analysis
- Brand perception monitoring
- Competitive product intelligence
- Customer sentiment tracking

---

# Project Impact

Proyek ini menunjukkan bagaimana opini pengguna di media sosial dapat diubah menjadi **product intelligence** yang mendukung pengambilan keputusan bisnis.

Beberapa capaian utama dari proyek ini:

• Menganalisis lebih dari **4.600 komentar pengguna YouTube** dari beberapa channel reviewer teknologi besar di Indonesia.

• Mengembangkan pipeline **Hybrid Sentiment Analysis + SetFit Fine-Tuning** untuk meningkatkan pemahaman konteks opini pengguna.

• Menggunakan **Active Learning Strategy** untuk meningkatkan kualitas dataset berlabel secara efisien tanpa perlu labeling seluruh data.

• Mengidentifikasi **pain points produk secara otomatis berdasarkan aspek fitur** seperti kamera, baterai, layar, dan software.

• Menggunakan **Z-Score analysis** untuk memprioritaskan fitur yang paling berdampak terhadap persepsi pengguna.

• Menghasilkan **AI-generated executive report menggunakan Gemini Pro API** untuk menerjemahkan insight data menjadi rekomendasi bisnis.

• Menyediakan **dashboard interaktif berbasis Streamlit** agar stakeholder dapat mengeksplorasi insight tanpa perlu menjalankan kode analisis.

---
## Methodology Overview

Proyek ini menggunakan pipeline Natural Language Processing (NLP) untuk mengubah komentar pengguna yang tidak terstruktur menjadi insight yang dapat digunakan dalam pengambilan keputusan bisnis:

1. **Data Mining**, Tahap ini bertujuan untuk memperoleh dataset opini pengguna yang realistis dan representatif terhadap persepsi pasar.
   - Komentar dikumpulkan dengan melakukan scraping menggunakan [YouTube Data API v3](https://developers.google.com/youtube/v3/getting-started) pada beberapa channel review gadget Indonesia yang memiliki jumlah komentar tinggi dan komunitas penonton aktif. Tahap ini bertujuan untuk memperoleh dataset opini pengguna yang **realistis** dan **representatif** terhadap persepsi pasar.

2. **Text Preprocessing**, Proses ini bertujuan untuk memastikan bahwa data yang digunakan oleh model memiliki kualitas yang lebih bersih dan konsisten.
   - Penghapusan emoji dan karakter tidak relevan
   - Normalisasi slang atau bahasa informal
   - Standardisasi teks agar konsisten
   - Filtering komentar kosong atau tidak bermakna

3. **Hybrid Sentiment Analysis**, Pendekatan hybrid ini dipilih agar model dapat menangkap baik konteks bahasa maupun sinyal domain spesifik dari komunitas pengguna gadget.
   - **IndoBERT sentiment classifier**, untuk memahami konteks kalimat secara mendalam
   - Keyword boosting, untuk memperkuat sinyal sentimen dari istilah atau frasa yang sering muncul dalam diskusi gadget

4. **Model Fine-Tuning**, Pendekatan ini memungkinkan model menjadi lebih spesifik terhadap domain analisis.
   - **SetFit (Sentence Transformer)**, Model ini dilatih menggunakan golden dataset berlabel manual sehingga model dapat lebih memahami pola bahasa, opini, dan ekspresi yang sering muncul dalam komentar pengguna YouTube.

5. **Active Learning**, digunakan untuk meningkatkan kualitas dataset berlabel secara efisien.
   - Model secara iteratif memilih komentar yang paling informatif untuk dilabel ulang oleh manusia. Strategi ini memungkinkan peningkatan performa model tanpa harus melakukan labeling pada seluruh dataset. 

6. **Aspect-Based Categorization**, Pendekatan ini memungkinkan analisis tidak hanya berhenti pada apakah sentimennya positif atau negatif, tetapi juga fitur produk mana yang menjadi sumber kepuasan atau keluhan pengguna.
   - Beberapa aspek yang dianalisis antara lain:
     - Camera
     - Battery
     - Display
     - Design
     - Software
     - Performance

7. **Z-Score Prioritization**, Untuk menentukan prioritas masalah atau kekuatan produk, digunakan pendekatan Z-Score analysis.
   - Metode ini membantu mengidentifikasi aspek yang memiliki sentimen secara statistik lebih tinggi atau lebih rendah dibandingkan rata-rata keseluruhan. Dengan pendekatan ini, analisis dapat mengungkap:
     - Fitur yang menjadi keunggulan utama produk
     - Fitur yang berpotensi menjadi pain point bagi pengguna

8. **Strategic Insight Generation**, Insight yang dihasilkan dari analisis data kemudian dirangkum menjadi laporan strategis menggunakan Large Language Model (LLM) melalui Gemini Pro API. Pendekatan ini membantu mengubah hasil analisis teknis menjadi rekomendasi bisnis yang lebih mudah dipahami oleh stakeholder.
   - Dengan memanfaatkan teknik prompt engineering, model AI menghasilkan laporan yang berisi:
     - Executive summary
     - Diagnosa persepsi pasar
     - Identifikasi risiko produk
     - Rekomendasi strategi produk dan marketing

9. **Interactive Dashboard**, Untuk mempermudah eksplorasi hasil analisis, dibangun dashboard interaktif menggunakan Streamlit. Dashboard ini berfungsi sebagai alat eksplorasi data yang dapat digunakan langsung oleh stakeholder tanpa perlu menjalankan kode analisis.
   - Dashboard ini memungkinkan pengguna untuk:
     - Menjelajahi distribusi sentimen
     - Menganalisis sentimen berdasarkan aspek produk
     - Memfilter komentar berdasarkan channel reviewer
     - Mengeksplorasi insight secara interaktif

---

# Training Infrastructure

Performa fine-tuning diuji pada beberapa environment cloud:

| Environment | Hardware | Training Time |
|-------------|----------|---------------|
| MacBook Air M3 | CPU | ~7 jam |
| Microsoft Azure VM | CPU (16 Core) | ~32 menit |
| Google Cloud VM | GPU (NVIDIA T4) | ~17 menit |

Benchmark ini menunjukkan peningkatan efisiensi training secara signifikan ketika menggunakan GPU.

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
