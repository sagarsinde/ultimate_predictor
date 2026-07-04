import json

with open('Prediction_Engine.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 1 to use %pip and add catboost
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for i, line in enumerate(cell['source']):
            if line.startswith('!pip install') or line.startswith('%pip install'):
                cell['source'][i] = '%pip install xgboost scikit-learn pandas numpy networkx catboost -q\n'

# Add v3 cells
v3_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. V3 ENGINE (CatBoost & DowFreq)\n",
        "Run the new v3 engine tests here!"
    ]
}

v3_kalyan_backtest = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /content/ultimate_predictor/\n",
        "!python -m v3.run_backtest kalyan"
    ]
}

v3_mb_backtest = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /content/ultimate_predictor/\n",
        "!python -m v3.run_backtest mb"
    ]
}

v3_kalyan_predict = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /content/ultimate_predictor/\n",
        "!python -m v3.run_predict kalyan"
    ]
}

v3_mb_predict = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "%cd /content/ultimate_predictor/\n",
        "!python -m v3.run_predict mb"
    ]
}

nb['cells'].extend([v3_markdown, v3_kalyan_backtest, v3_mb_backtest, v3_kalyan_predict, v3_mb_predict])

with open('Prediction_Engine.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully!")
