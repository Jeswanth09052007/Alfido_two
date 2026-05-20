# House Price Regression Project

## Goal
Build a regression model to predict house prices using feature engineering, missing-value handling, and model selection.

## Dataset
The dataset is stored at `data/house_prices.csv`.

## What is included
- EDA and visualizations
- Feature transformations: log transform, encoding, scaling
- Missing-value handling using `SimpleImputer`
- Models compared:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluation metrics: RMSE, MAE, R2
- Residual analysis plots
- Saved best model in `models/best_house_price_model.joblib` and `.pkl`
- Example inference code in `src/inference.py`

## Best Model
Best model selected by lowest RMSE: **Linear Regression**

## Metrics

| Model             |   RMSE |      MAE |       R2 |
|:------------------|-------:|---------:|---------:|
| Linear Regression | 173370 |  87261.2 | 0.797962 |
| Random Forest     | 220447 | 111008   | 0.673342 |
| Gradient Boosting | 239498 | 124603   | 0.614444 |

## How to run

```bash
pip install -r requirements.txt
python src/train_model.py
python src/inference.py
```

## Notebook
Open this notebook:

```text
notebooks/House_Price_Regression.ipynb
```

Run all cells from top to bottom.

## Submission Rules
For the task submission page, submit links clearly with the task number:

1. GitHub repository link containing notebook, source code, README, requirements, and plots.
2. Google Drive link for large files or final ZIP, if required.
3. Notebook link.
4. Model file link.
5. Submission DOC/PDF from `docs/` folder.

Before submitting, replace placeholder links in the DOC/PDF with your actual GitHub and Drive links.
