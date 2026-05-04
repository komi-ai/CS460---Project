# CS460---Project

Movie recommendation project using MovieLens data and multiple recommendation models.


## Project Structure

- [recommender.py](recommender.py): main script for downloading data, training models, and saving results.
- [data](data): MovieLens datasets.
- [results/results.json](results/results.json): saved evaluation metrics for completed runs.

## How to Run

1. Activate your virtual environment.
2. Install the required Python packages if they are not already installed.
3. Run the script with the datasets and models you want to evaluate.

Example:

```bash
python recommender.py --datasets 100k 1m --models knn svd als hybrid
```

This writes the output metrics to [results/results.json](results/results.json).


