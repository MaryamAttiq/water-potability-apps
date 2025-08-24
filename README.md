# Water Potability – Streamlit App

## Train (with class balancing)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# Put water_potability.csv in this folder first
python train_model.py --csv water_potability.csv --sampling over --n_iter 30 --scoring f1 --out_dir outputs
```

This creates:
- `outputs/model.pkl` – trained pipeline (imputer + sampler + RandomForest)
- `outputs/metrics.json` – test scores
- `outputs/confusion_matrix.png`
- `outputs/feature_importances.csv`

## Run app locally
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push these files to a GitHub repo.
2. Go to https://share.streamlit.io → New app → Select your repo, branch, and `app.py`.
3. Ensure `requirements.txt` and `outputs/model.pkl` are in the repo.
4. Deploy.
```
