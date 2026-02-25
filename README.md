# Xiaomi Sentiment Intelligence System
End to end Sentiment Intelligence platform for analyzing YouTube product reviews of Xiaomi Note 15 Pro.
*Hybrid ML + Active Learning + Aspect-Based Intelligence + Executive LLM Reporting.*

---
### This project combines:
- Hybrid Baseline (IndoBERT + Keyword Boosting)
- SetFit Fine-Tuning
- Active Learning (Multi-Round)
- Aspect-Based Intelligence
- Business Prioritization Modeling
- Executive LLM Report Generation
- Interactive Streamlit Dashboard

### **Installation**
1. Create Environment:
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
streamlit run app.py
```
