# BIP_Gruppe_1
# 📚 Project Title

Short summary of the project in 2–3 sentences.  
What does it do? Why is it useful? What is the goal?

---

## 📖 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Analyses & Results](#analyses--results)
- [Known Issues / Limitations](#known-issues--limitations)
- [Future Work](#future-work)
- [Contributors](#contributors)
- [License](#license)

---

## 🔍 Overview
Explain the project background and objectives.  

This project analyzes museum objects from the Technisches Museum Berlin based on  JPG images.  
The aim is to test the feasibility of automatically generating object descriptions and integrating them into a data pipeline.

---

## ✨ Features
- Importing and cleaning inconsistent Excel metadata  
- Searching and counting images for each object  
- Analyzing distributions (e.g., number of images per object, manufacturing years)  
- Generating automatic image descriptions (BLIP, InstructBLIP, etc.)  
- Exporting results into CSV or visualizations (histograms, boxplots)  
- using an AI-API to generate Dexcription  of objects

---

## 📊 Data
- **Excel/CSV File** (`fertig.csv`): contains object metadata  
  - Example: object number, description, year of manufacture  
- **Image Folders**: contain JPG images of objects, structured by object number  
- Note: Some objects have few or no images, and descriptions are inconsistent.  
  These are important findings for potential model training and make direct pipeline integration challenging.  

---

## ⚙️ Installation
Requirements:
- Python 3.9+  
- Jupyter Notebook  
- Required libraries:  
  ```bash
  pip install -r requirements.txt
