# 🧱 PIV – LEGO Piece Detection & Classification

> Practical assignment for the course **Processamento de Imagem e Visão (PIV)**  
> BSc in Computer Engineering and Multimedia – **ISEL**  
> Winter semester **2025 / 2026**

---

## 👥 Authors

- **Name:** Miguel Cordeiro — nº 49765 — LEIM51N  
- **Name:** Bruno Santos — nº 45096 — LEIM51N  
- **Instructor:** Eng. Nuno Silva  

---

## 📝 Project Overview

This project implements a complete image-processing pipeline in Python to **detect, segment and classify LEGO bricks** from RGB images.

Starting from raw images of LEGO pieces on a table, the system:

1. Analyses the input images (colour channels, histograms, dataset exploration)  
2. Applies **binarization** to separate pieces from the background  
3. Uses **morphological operations** to clean and refine the binary mask  
4. Extracts **connected components** (one component ≈ one LEGO brick)  
5. Computes **features** (mainly area) for each component  
6. Classifies each object into one of several **LEGO classes** (2x2, 2x4, 2x6, 2x8) plus a **rejection class**

All development and reporting are contained in the Jupyter Notebook:

```text
notebooks/P1_51N_A49765_45096.ipynb
