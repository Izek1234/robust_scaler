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
