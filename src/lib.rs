use ndarray::{Array1, Array2};
use serde::{Deserialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// RobustScaler: A Rust implementation compatible with scikit-learn's RobustScaler.
/// Centers data using the median and scales using the interquartile range (IQR).
/// Resistant to outliers.
pub struct RobustScaler {
    center_: Vec<f64>,  // Median of each feature
    scale_: Vec<f64>,   // IQR (Q3 - Q1) of each feature
}

impl RobustScaler {
    /// Creates a new empty RobustScaler.
    pub fn new() -> Self {
        Self {
            center_: vec![],
            scale_: vec![],
        }
    }

    /// Fits the scaler to 2D data: computes median and IQR for each feature.
    ///
    /// # Arguments
    /// * `data` - A 2D array where rows are samples and columns are features.
    ///
    /// # Returns
    /// &Self for chaining.
    pub fn fit(&mut self, data: &Array2<f64>) -> &Self {
        let n_features = data.ncols();
        self.center_.clear();
        self.center_.reserve(n_features);
        self.scale_.clear();
        self.scale_.reserve(n_features);

        for i in 0..n_features {
            let feature = data.column(i);
            let center = median(&feature.to_owned());
            let q1 = quantile(&feature.to_owned(), 0.25);
            let q3 = quantile(&feature.to_owned(), 0.75);
            let iqr = (q3 - q1).max(1e-8); // Avoid division by zero

            self.center_.push(center);
            self.scale_.push(iqr);
        }

        self
    }

    /// Transforms 2D data using the fitted center and scale.
    ///
    /// # Arguments
    /// * `data` - Input 2D array to scale.
    ///
    /// # Returns
    /// Scaled `Array2<f64>`: (X - center) / scale
    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        assert_eq!(data.ncols(), self.center_.len());
        assert_eq!(data.ncols(), self.scale_.len());

        let mut result = data.clone();
        for (i, (center, scale)) in self.center_.iter().zip(&self.scale_).enumerate() {
            result.column_mut(i).mapv_inplace(|x| (x - center) / scale);
        }
        result
    }

    /// Transforms a 1D input vector (commonly used in APIs).
    ///
    /// # Arguments
    /// * `input` - Slice of feature values (length must match `n_features`).
    ///
    /// # Returns
    /// A `Vec<f32>` with scaled values.
    pub fn transform_1d(&self, input: &[f64]) -> Vec<f32> {
        if input.len() != self.center_.len() {
            panic!("Input size {} does not match scaler n_features {}", input.len(), self.center_.len());
        }

        input
            .iter()
            .zip(&self.center_)
            .zip(&self.scale_)
            .map(|((&x, &center), &scale)| ((x - center) / scale) as f32)
            .collect()
    }

    /// Fits the scaler and transforms the data in one step.
    ///
    /// # Arguments
    /// * `data` - Input 2D array.
    ///
    /// # Returns
    /// Scaled `Array2<f64>`.
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Array2<f64> {
        self.fit(data);
        self.transform(data)
    }

    /// Loads a pre-trained RobustScaler from a JSON file (exported from scikit-learn).
    ///
    /// Expects a JSON with:
    /// - "center_": list of medians
    /// - "scale_": list of IQRs
    /// - "n_features_in_": number of features
    ///
    /// # Arguments
    /// * `path` - Path to the JSON file.
    ///
    /// # Returns
    /// `Ok(RobustScaler)` if successful, `Err` otherwise.
    pub fn from_json<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);

        let params: SklearnRobustScalerParams = serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if params.center.len() != params.n_features_in {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Length of 'center_' does not match 'n_features_in_'",
            ));
        }
        if params.scale.len() != params.n_features_in {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Length of 'scale_' does not match 'n_features_in_'",
            ));
        }

        Ok(Self {
            center_: params.center,
            scale_: params.scale,
        })
    }

    /// Returns the number of features the scaler was trained on.
    pub fn n_features(&self) -> usize {
        self.center_.len()
    }
}

// --- Helper functions ---

/// Computes the median of a 1D array.
fn median(data: &Array1<f64>) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Computes the quantile of a 1D array using linear interpolation.
///
/// # Arguments
/// * `data` - Input array.
/// * `q` - Quantile value (0.0 ≤ q ≤ 1.0).
fn quantile(data: &Array1<f64>, q: f64) -> f64 {
    assert!(q >= 0.0 && q <= 1.0);
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len() as f64;
    let index = q * (n - 1.0);
    let i = index.floor() as usize;
    let t = index - i as f64;

    if i >= sorted.len() - 1 {
        sorted[sorted.len() - 1]
    } else {
        sorted[i] + t * (sorted[i + 1] - sorted[i])
    }
}

// Parameters deserialized from scikit-learn's JSON
#[derive(Deserialize)]
struct SklearnRobustScalerParams {
    #[serde(rename = "center_")]
    center: Vec<f64>,

    #[serde(rename = "scale_")]
    scale: Vec<f64>,

    #[serde(rename = "n_features_in_")]
    n_features_in: usize,
}

// --- Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_transform_1d() {
        let mut scaler = RobustScaler::new();
        let data = arr2(&[
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        scaler.fit(&data);

        let input = vec![1.0, 2.0];
        let scaled = scaler.transform_1d(&input);
        // For first feature: median = 3, Q1 = 1, Q3 = 5 → IQR = 4 → (1-3)/4 = -0.5
        // But with only 3 points, quantiles may vary slightly
        assert!((scaled[0] - (-0.5) as f32).abs() < 1e-5);
    }
}
