# Automatic Museum Object Descriptions

This project explores the feasibility of **generating automatic catalogue-style descriptions for museum objects** using **generative AI**.  
The context is the Technisches Museum Berlin, where we were provided with several hundred object images and Excel metadata.

---

## 📌 Project Idea

The goal is to create **structured and human-readable descriptions** for museum objects:

- **Input:** multiple images of one object + metadata from Excel files  
- **Processing:** search, selection, and preparation of images, generation of captions with AI  
- **Output:**  
  - Structured CSV file with relevant fields  
  - Human-readable text descriptions (TXT)  
  - Both available in **English and German**  

This prototype evaluates both **local AI models** (running on the machine) and **API-based AI models** (via OpenAI).

---

## 🗂 Project Structure

- **`preVersions/`**  
  Contains earlier prototypes:  
  - Data analysis and preprocessing of Excel sheets  
  - Local model experiments (BLIP, ViT-GPT2)  
  - First API attempts  

- **`catalog/`**  
  Finalized and cleaned implementation of the pipeline.  
  Contains configuration, prompt design, LLM logic, and utilities to process object images into catalogue-ready outputs.  

- **`testRun.py`**  
  Entry point for the current version.  
  - Configure input folders (where object images are stored)  
  - Define which object(s) should be processed (via object number prefix)  
  - Set language (German or English)  
  - Limit maximum number of images per object  

- **Outputs**  
  - `catalog_descriptions.txt` → human-readable descriptions  
  - `descriptions_long.csv` → structured CSV in long format  

---

## ⚙️ Usage

1. **Open `testRun.py` and set your parameters** (edit the config block at the top of the file):
   ```python
   # testRun.py — user configuration
   ROOT_DIR    = "/path/to/images"          # root folder containing all images
   DEST_DIR    = "/path/to/output"          # where matched/copied images and results go
   PREFIXES    = ["1-2024-0062"]            # list of object-number prefixes to process
   LANG        = "de"                        # "de" or "en"
   MAX_IMAGES  = 12                          # max images per object sent to the model
   OUT_TXT     = "catalog_descriptions.txt"  # TXT output
   OUT_CSV     = "descriptions_long.csv"     # CSV (long format) output
### 🔑 Set your OpenAI API key

The script uses the **OpenAI API** to generate object descriptions.  
For this to work, you need a valid **API key** from [OpenAI](https://platform.openai.com/).  

1. In **PyCharm**, go to **Run → Edit Configurations…**  
2. Select your `testRun.py` run configuration (or create a new one).  
3. Under **Environment variables**, add a new entry:  


## ⚙️ Parameters

- `--root`: root folder with images  
- `--dest`: destination folder for matched/copied images  
- `--prefix`: object number prefix used to search for images  
- `--lang`: output language (`de` or `en`)  
- `--max-images`: maximum number of images per object included in the request  
- `--out-txt`: TXT output path (default: `catalog_descriptions.txt`)  
- `--out-csv`: CSV output path (default: `descriptions_long.csv`)  

The script generates both **TXT and CSV** outputs.  

---

## 🔎 Analyses & Insights

- Some objects contain **no or very few images**, others have multiple detailed perspectives.  
- Metadata is often **inconsistent** (years, year ranges like `1950–1960`, textual notes).  
- Such inconsistencies are **critical for training custom models** and complicate integration into pipelines.  
- Combining multiple images per object leads to more robust descriptions.  

---

## 🚧 Known Issues / Limitations

- Local models produce limited quality due to computational constraints.  
- API-based models achieve higher quality but require sending data to external servers.  
- Metadata requires heavy preprocessing.  
- Current version is a **prototype** and not production-ready.  

---

## 🔮 Future Work

- Explore additional LLMs (local & API).  
- Improve image preprocessing (cropping, contrast, background removal).  
- Build a **local inference server** for sensitive data.  
- Multi-language expansion (EN/DE, more if needed).  
- Create metrics to evaluate the quality of generated descriptions.  

---

## 👥 Contributors

- David 
- Ismail
- Group BIP_1  

---

## 📜 License

This project is developed as a **prototype for research and educational purposes**.  
We decided to release the code under a **permissive license – everyone may use and adapt it**.  
Please ensure that you respect usage rights of included **images and metadata** provided by the Technisches Museum Berlin.  
---

## 💡 Acknowledgements

All project ideas, concepts, and methodological approaches were developed by our team.  
For the **implementation and coding process**, we also made use of **AI-based coding assistants** and other supportive tools.  
These were applied solely to **accelerate development** and improve efficiency, while the overall design, workflow, and decision-making originated from us.  
