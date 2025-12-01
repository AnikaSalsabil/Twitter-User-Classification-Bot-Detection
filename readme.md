# Assignment –2: Twitter User Dataset Analysis for Human vs Non-Human Classification

This repository contains the codes for analyzing the Twitter user dataset (human vs brand) across preprocessing, EDA, association rules, clustering, classification, and text processing.

## Project Structure

To run the `.py` files successfully, ensure the following setup:

-   Place the main data file **twitter_user_data.csv** at the **root** level (same location as the `.py` files).
-   An `out` folder is created automatically by the preprocessing script; all results are saved there.

root/
│── out/ # generated after running scripts, stores results
│── preprocessing.py
│── EDA.py
│── association_rules.py
│── clustering.py
│── classification_regression.py
│── text_processing.py

## Execution Order

1. `preprocessing.py`
2. `EDA.py`
3. `association_rules.py`
4. `clustering.py`
5. `classification_regression.py`
6. `text_processing.py`

## Additional Notes

-   Some scripts may sample or cache intermediate artifacts for speed and reproducibility.
-   Figures produced by EDA can be configured to save under `out/figures` to keep all outputs in one place.

⚠️ Runtime Note  
The scripts may take **2–5 minutes** to run depending on your machine, as they include TF-IDF, association rule mining, clustering, and model training.

---

## Packages to install (Installation Commands)

1. pip install numpy
2. pip install pandas
3. pip install scipy
4. pip install scikit-learn
5. pip install matplotlib
6. pip install seaborn
7. pip install wordcloud
8. pip install pillow
9. pip install nltk
10. pip install gensim
11. pip install minisom
12. pip install mlxtend
