# %% [markdown]
# # Galaxy Image to Redshift Prediction System
# 
# This notebook implements an end-to-end pipeline that:
# 1. Takes galaxy images as input
# 2. Extracts photometric features from the images
# 3. Uses a Random Forest model to predict redshift
# 4. Provides error estimation
# 
# The workflow follows:
# 
# ```
# Galaxy Image
#     │
#     ▼
# Image Processing
#     │
#     ▼
# Feature Extraction
#     │
#     ▼
# Feature Matrix
#     │
#     ▼
# Random Forest Model
#     │
#     ▼
# Redshift Prediction
#     │
#     ▼
# Error Estimation
# ```

# %%
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.aperture import aperture_photometry, CircularAperture
import cv2
import os
import joblib
import requests
from io import BytesIO

# Setup visualization
plt.style.use('seaborn-darkgrid')
sns.set_palette("husl")
# %matplotlib inline

# %% [markdown]
# ## 1. Image Processing and Feature Extraction

# %%
def process_galaxy_image(image_path):
    """
    Process a galaxy image and extract photometric features
    Returns a dictionary of features
    """
    # Load image
    if isinstance(image_path, str):
        if image_path.startswith('http'):
            response = requests.get(image_path)
            img = np.frombuffer(BytesIO(response.content).read())
        else:
            img = fits.getdata(image_path)
    else:
        img = image_path
    
    # Convert to 2D if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    # Basic image processing
    img = img.astype(np.float32)
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    img -= median  # Background subtraction
    
    # Calculate image statistics
    height, width = img.shape
    y, x = np.indices(img.shape)
    
    # Calculate photometric features
    total_flux = np.sum(img)
    max_intensity = np.max(img)
    centroid_x = np.sum(x * img) / total_flux
    centroid_y = np.sum(y * img) / total_flux
    
    # Aperture photometry
    positions = [(centroid_x, centroid_y)]
    apertures = [CircularAperture(positions, r=r) for r in [5, 10, 15]]
    phot_table = aperture_photometry(img, apertures)
    
    # Feature extraction
    features = {
        'total_flux': total_flux,
        'max_intensity': max_intensity,
        'centroid_x': centroid_x,
        'centroid_y': centroid_y,
        'flux_ratio_10_5': phot_table['aperture_sum_1'] / phot_table['aperture_sum_0'],
        'flux_ratio_15_5': phot_table['aperture_sum_2'] / phot_table['aperture_sum_0'],
        'asymmetry': calculate_asymmetry(img),
        'concentration': phot_table['aperture_sum_2'] / phot_table['aperture_sum_0'],
        'size': calculate_size(img),
        'ellipticity': calculate_ellipticity(img),
        'variance': np.var(img)
    }
    
    return features

def calculate_asymmetry(image):
    """Calculate galaxy asymmetry index"""
    rotated = np.rot90(image, 2)
    residual = np.abs(image - rotated)
    return np.sum(residual) / np.sum(np.abs(image))

def calculate_size(image):
    """Calculate effective radius"""
    threshold = 0.5 * np.max(image)
    return np.sqrt(np.sum(image > threshold) / np.pi)

def calculate_ellipticity(image):
    """Calculate galaxy ellipticity"""
    y, x = np.indices(image.shape)
    cy, cx = np.array(image.shape) / 2
    
    # Calculate second moments
    mxx = np.sum((x - cx)**2 * image) / np.sum(image)
    myy = np.sum((y - cy)**2 * image) / np.sum(image)
    mxy = np.sum((x - cx) * (y - cy) * image) / np.sum(image)
    
    # Calculate ellipticity
    e1 = (mxx - myy) / (mxx + myy)
    e2 = 2 * mxy / (mxx + myy)
    return np.sqrt(e1**2 + e2**2)

# %% [markdown]
# ## 2. Prepare Dataset from Images

# %%
def create_dataset_from_images(image_folder, metadata_csv):
    """
    Create dataset from galaxy images and metadata
    Returns DataFrame with features and redshift
    """
    # Load metadata
    metadata = pd.read_csv(metadata_csv)
    dataset = []
    
    # Process each image
    for idx, row in metadata.iterrows():
        try:
            img_path = os.path.join(image_folder, row['image_filename'])
            features = process_galaxy_image(img_path)
            features['redshift'] = row['redshift']
            features['objid'] = row['objid']
            dataset.append(features)
        except Exception as e:
            print(f"Error processing {row['image_filename']}: {str(e)}")
    
    return pd.DataFrame(dataset)

# %% [markdown]
# ## 3. Load and Preprocess Data

# %%
# Create dataset from images (this would take time for real images)
# For demo purposes, we'll use a pre-existing CSV
# Uncomment to process actual images:
# df = create_dataset_from_images('galaxy_images/', 'data.csv')

# Instead, we'll load the provided CSV
df = pd.read_csv('data.csv')

# Clean and prepare data
print("Original shape:", df.shape)

# Handle missing values
df.replace(-9999.0, np.nan, inplace=True)
df.dropna(inplace=True)

print("Cleaned shape:", df.shape)

# Visualize data
plt.figure(figsize=(12, 8))
sns.pairplot(df[['total_flux', 'concentration', 'asymmetry', 'redshift']].sample(1000))
plt.suptitle('Feature Relationships', y=1.02)
plt.show()

# %% [markdown]
# ## 4. Train Random Forest Model

# %%
# Prepare features and target
target = 'redshift'
features = ['total_flux', 'max_intensity', 'flux_ratio_10_5', 
            'flux_ratio_15_5', 'asymmetry', 'concentration', 
            'size', 'ellipticity', 'variance']

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, 'image_to_redshift_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Feature importance
importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importance.values, y=importance.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Redshift Prediction and Error Estimation

# %%
# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_squared_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Visualization
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.title('Actual vs Predicted Redshift')
plt.xlabel('Actual Redshift')
plt.ylabel('Predicted Redshift')
plt.grid(True)
plt.show()

# Residual plot
residuals = y_pred - y_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted Redshift')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. Complete Image-to-Redshift Pipeline

# %%
class GalaxyRedshiftPredictor:
    def __init__(self, model_path='image_to_redshift_model.pkl', scaler_path='feature_scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.features = [
            'total_flux', 'max_intensity', 'flux_ratio_10_5', 
            'flux_ratio_15_5', 'asymmetry', 'concentration', 
            'size', 'ellipticity', 'variance'
        ]
    
    def predict_from_image(self, image_path):
        """Process image and predict redshift"""
        # Extract features
        features = process_galaxy_image(image_path)
        
        # Create feature array
        feature_array = np.array([features[f] for f in self.features]).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(feature_array)
        
        # Predict redshift
        redshift = self.model.predict(scaled_features)[0]
        
        # Estimate error (simplified)
        error = 0.05 + 0.1 * redshift  # Example error model
        
        return {
            'redshift': redshift,
            'error': error,
            'features': features
        }

# %% [markdown]
# ## 7. Example Usage

# %%
# Initialize predictor
predictor = GalaxyRedshiftPredictor()

# Example prediction (using a sample image)
# In practice, this would be a path to a real galaxy image
sample_image = "https://www.sdss4.org/wp-content/uploads/2015/09/galaxy_M51-300x300.jpg"

# Predict redshift
result = predictor.predict_from_image(sample_image)

# Display results
print("\nPrediction Results:")
print(f"Redshift: {result['redshift']:.4f} ± {result['error']:.4f}")
print("\nExtracted Features:")
for feature, value in result['features'].items():
    print(f"{feature:>20}: {value:.4f}")

# Visualize image
if isinstance(sample_image, str) and sample_image.startswith('http'):
    response = requests.get(sample_image)
    img = plt.imread(BytesIO(response.content))
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Predicted Redshift: {result['redshift']:.4f} ± {result['error']:.4f}")
    plt.axis('off')
    plt.show()