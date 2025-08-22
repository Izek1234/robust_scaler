# robust_scaler

A Rust implementation of scikit-learn's `RobustScaler`, designed for robust feature scaling in machine learning pipelines.

`RobustScaler` removes the median and scales data according to the interquartile range (IQR), making it robust to outliers.

This crate is ideal for preprocessing input data in ML services, especially when integrating with models trained in Python and deployed in Rust.

## 🚀 Features

- ✅ Compatible with scikit-learn's `RobustScaler`
- ✅ Load from JSON (exported from `sklearn`)
- ✅ 1D and 2D data support
- ✅ No external dependencies beyond `ndarray` and `serde`
- ✅ Thread-safe usage with `Lazy` or `Arc`
- ✅ Perfect for web APIs (e.g., Actix, Axum)

## 📦 Installation

Add to your `Cargo.toml`:

```toml
robust_scaler = "0.1.0"
```

Or use locally during development:

```toml
robust_scaler = { path = "./robust_scaler" }
```

## 🔧 Usage

### Load from JSON (recommended for production)

```rust
use robust_scaler::RobustScaler;

let scaler = RobustScaler::from_json("robust_scaler.json")
    .expect("Failed to load scaler");

let scaled = scaler.transform_1d(&[1.0, 2.0, 3.0]);
```

The JSON should be exported from scikit-learn like this:

```python
import json
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaler.fit(data)

with open("robust_scaler.json", "w") as f:
    json.dump({
        "center_": scaler.center_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "n_features_in_": scaler.n_features_in_
    }, f)
```

### Fit and transform in Rust (for testing)

```rust
use ndarray::arr2;
use robust_scaler::RobustScaler;

let data = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
let mut scaler = RobustScaler::new();
scaler.fit(&data);

let scaled = scaler.transform(&data);
```
