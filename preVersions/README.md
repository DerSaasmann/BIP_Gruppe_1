# Automatic Museum Object Descriptions

This project explores the feasibility of **generating automatic description texts for museum objects** using generative AI models.  
The context is the Technisches Museum Berlin, where we were provided with several hundred object images as well as Excel tables with additional metadata.

---

## 📌 Project Idea
The goal is to automatically create **catalogue-style descriptions** for museum objects:
- Input: multiple images of an object + metadata from Excel files
- Processing: cleaning and preparing metadata, grouping and selecting relevant images
- Output: structured JSON/CSV files and human-readable text descriptions

This prototype investigates both **local AI models** (running on the machine) and **API-based AI models** (using OpenAI’s API) to evaluate quality, performance, and feasibility.

---

## 🗂 Project Structure

- **`DataAnalyze.ipynb` & `DataProcessing.ipynb`**  
  Jupyter Notebooks used to combine and clean the Excel tables, remove inconsistencies, and derive insights (e.g., missing images, inconsistent descriptions).  

- **`runninglocal.py`, `runninglocalBLIP.py`, `evenBetterLocalModel.py`**  
  Experiments with running local image captioning models (e.g., BLIP, ViT-GPT2) to analyze images without sending data to external servers.  
  ➡️ Result: technically functional, but limited output quality due to restricted computing power.  

- **`beschreibe_bilder.py` & `v2_beschreibe_bilder.py`**  
  Scripts for API-based analysis using OpenAI models.  
  These scripts collect all images belonging to one object and send them in a single API request. The model then returns structured JSON with fields relevant for museum cataloguing (classification, features, free-text description).  

- **`BilderFinden.ipynb`**  
  Helper notebook to search and filter images belonging to a given object number and prepare them for further processing.  

- **Data files (`fertig.csv`, `Schreibmaschinen.csv`, `Liste_AK Kommunikation.csv`, `nummern_clean.csv`)**  
  Contain object metadata and cleaned IDs used to link Excel information with image files.  

- **Outputs (`catalog_descriptions.txt`, `descriptions_long.csv`)**  
  Results of the AI analysis: text summaries and long-format CSVs with structured fields.  

---

## 🔎 Analyses & Insights
During the data analysis phase we found:
- Some objects have **few or no images**, while others have multiple views.  
- **Descriptions are highly inconsistent** across the Excel sheets (sometimes just a year, sometimes year + text, sometimes ranges like `1950–1960`).  
- These inconsistencies are **important to consider if one wants to train a custom model**.  
- They also make it harder to integrate the data into a uniform processing pipeline.  

---

## ⚙️ Usage

1. Place all images of one object in a dedicated folder (`DEST_DIR` in `v2_beschreibe_bilder.py`).  
2. Run the script:  

   ```bash
   python v2_beschreibe_bilder.py
The script sends all images together to the AI model and generates:

- **Text output** (`catalog_descriptions.txt`)  
- **Structured CSV output** (`descriptions_long.csv`)  

For local experiments (without API), run one of the `runninglocal*.py` scripts.

---

## 🚧 Known Issues / Limitations

- Local models produce limited quality due to computational constraints.  
- API-based models (OpenAI) achieve better quality but require sending data to external servers.  
- Metadata (Excel) is inconsistent and requires heavy preprocessing.  
- Current version is a **prototype** and not yet production-ready.  

---

## 🔮 Future Work

- Test additional LLMs (local and cloud-based).  
- Improve image preprocessing (cropping, contrast, background removal).  
- Set up a local inference server for privacy-sensitive data.  
- Extend support for multiple languages (German/English).  
- Create evaluation metrics to measure the quality of generated descriptions.  

---

## 👥 Contributors

- David Assmann  
- Group BIP_1  

---

## 📜 License

This project is developed as a **prototype for research and educational purposes**.  
Please check usage rights of included data (images, metadata) with the Technisches Museum Berlin.  