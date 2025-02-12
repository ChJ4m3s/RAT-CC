# RAT-CC implementation

This repository is a Python implementation of the RAT-CC model.

## Setup

1. Create virtual env: `python -m venv venv`;
2. Activate the virtual environment: `source venv/bin/activate`;
3. Install requirements: `pip install -r requirements.txt`.

## Overview

The repository defines the `rat_cc` function that returns an instance of the RAT-CC model. The required input parameters are:

- `shape`: The shape of the original data;
- `emb_size`: The size of the embedding (the compressed representation);
- `n_layers`: The number of hidden layers for the recurrent autoencoder (note that the encoder and the dencoder have the same number of hidden layers);
- `units`: The dimensionality of the output space for hidden layers;
- `l`: The lambda value.

## Example usage

```python
# Instantiate the model
model = rat_cc((1000, 200, 2), 80, 5, 50)

# Train the model with training dataset
model.fit(
  x_train,
  x_train,
  validation_split=0.2,
)

# Get predictions
reconstructed, embedding = model.predict(x_test)
```
