# User Identity and Product Recommendation System - Team Contributions Report

**Project**: Multimodal Data Preprocessing 
**Course**: Machine Learning Pipeline  
**Team**: Group 11  
**GitHub Repository**: [https://github.com/excelasaph/Data-Preprocessing-Group-11](https://github.com/excelasaph/Data-Preprocessing-Group-11)  
**Demo Video**: [https://youtube.com/watch?v=YOUR_VIDEO_ID](https://youtube.com/watch?v=YOUR_VIDEO_ID)

---

## Executive Summary

This report details the implementation of a multimodal biometric security system that integrates facial recognition, voiceprint verification, and personalized product recommendations. The project fulfills all requirements across four main tasks, with each team member contributing significantly to different aspects of the system.

### Project Overview
- **Data Collection**: 4 team members × 3 expressions + 2 audio phrases with augmentation
- **Model Development**: 3 machine learning models (facial recognition, voice verification, product recommendation)
- **System Integration**: Demo with real-time authentication and unauthorized access simulation
- **Feature Engineering**: Automated pipeline for image and audio feature extraction

### Model Metrics
#### Product Recommendation Model:
- **Accuracy**: 65% 
- **F1-Score** - 61%
- **Loss** - 0.8349

#### Facial Recognition Model:
- **Accuracy**: 90% 
- **F1-Score** - 90%
- **Loss** - 0.6440

#### Voiceprint Verification Model:
- **Accuracy**: 85.71%
- **F1-Score** - 67%
- **Loss** - 0.5188

### Overall Evaluation Metrics
- **Accuracy**: Face (90%), Voice (85.71%), Product (65%)
---

## Team Member Contributions

### 1. Anne Marie Twagirayezu - Image Processing & Facial Recognition Model

#### Task 1: Image Data Collection and Processing

**Data Collection Pipeline:**
- Collected 12 original images (4 team members × 3 expressions)
- Implemented image augmentation pipeline
- Created image processing workflow

**Image Augmentation Implementation:**
```python
# Image augmentation pipeline
def augment_image(image, size=(64, 64)):
    image = clean_image(image, size)
    flipped = clean_image(cv2.flip(image, 1), size)
    rotated = clean_image(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), size)
    gray = clean_image(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), size)
    return [image, flipped, rotated, gray]
```

**Feature Extraction Development:**
- Implemented color histogram feature extraction
- Created 512-dimensional feature vectors (8×8×8 RGB bins)
- Developed automated feature extraction pipeline
- Generated `image_features.csv` with 48 samples (12 originals × 4 augmentations)

**Facial Recognition Model Training:**
- Implemented XGBoost classifier for facial recognition
- Achieved 90% accuracy on test set
- Created model persistence and loading procedures
- Developed confidence-based authentication system

**Files Created/Modified:**
- `Data/pictures/` - Original image collection
- `augmented/` - Augmented image storage
- `Datasets/image_features.csv` - Extracted image features
- `models/facial_recognition_xgboost_model.joblib` - Trained model
- `Notebooks/Image_processing&_Facial_recognition_model.ipynb` - Processing notebook

---

### 2. Excel Asaph - Data Merging & Product Recommendation Model

#### Task 2: Data Merging and Product Recommendation System

**Data Merging Implementation:**
- Merged `customer_social_profiles.csv` and `customer_transactions.csv`
- Implemented data cleaning and preprocessing
- Created feature engineering pipeline for the merged customer data

**Data Preprocessing Pipeline:**
```python
# Data merging and feature engineering
def merge_customer_data(social_profiles, transactions):
    # Handle duplicates and null values
    social_clean = social_profiles.groupby('customer_id_new').agg({
        'engagement_score': 'mean',
        'purchase_interest_score': 'mean',
        'review_sentiment': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        'social_media_platform': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    }).reset_index()
    
    # Clean transactions data
    transactions_clean = transactions.groupby('customer_id_legacy').agg({
        'transaction_id': 'count',
        'purchase_amount': 'sum',
        'purchase_date': 'first',
        'product_category': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],
        'customer_rating': 'mean'
    }).reset_index()
    
    # Merge datasets
    merged_data = pd.merge(social_clean, transactions_clean, 
                          left_on='customer_id', 
                          right_on='customer_id_legacy', 
                          how='left')
    
    return merged_data
```

**Feature Engineering:**
- Created 15 engineered features for product recommendation
- Implemented categorical encoding (one-hot encoding)
- Developed temporal feature extraction (`purchase_date` encoding)
- Created engagement and purchase pattern metrics

**Product Recommendation Model Training:**
- Implemented **Random Forest** with **GridSearchCV** hyperparameter tuning
- Used parameter grid: n_estimators=[100,200,300], max_depth=[None,10,20,30] to get the best parameters for the training data
- Achieved 65% accuracy on test set
- Best parameters: max_depth=None, min_samples_leaf=2, min_samples_split=2, n_estimators=300
- Created model evaluation and feature importance analysis

**Files Created/Modified:**
- `Data/datasets/` - Original customer datasets
- `Datasets/merged_customer_data.csv` - Merged dataset
- `models/product_recommendation_model.pkl` - Trained model
- `encoders/product_recommendation_scaler.pkl` - Feature scaler
- `Notebooks/Data_Merging_Product_Recommendation_Model_Training.ipynb` - Processing notebook

---

### 3. Christophe Gakwaya - Audio Processing & Voiceprint Verification Model

#### Task 3: Audio Data Collection and Processing

**Audio Data Collection:**
- Collected 8 original audio samples (4 team members × 2 phrases)
- Implemented audio augmentation pipeline
- Created audio processing workflow

**Audio Augmentation Implementation:**
```python
# Audio augmentation pipeline
def augment_audio(y, sr):
    # Pitch shift (±2 semitones)
    y_pitch_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    
    # Time stretch (1.2x speed)
    y_time_stretched = librosa.effects.time_stretch(y, rate=1.2)
    
    # Noise addition (0.5% amplitude)
    noise = np.random.randn(len(y))
    noise_amplitude = 0.005 * np.max(np.abs(y))
    y_noisy = y + noise_amplitude * noise
    
    return [y, y_pitch_shifted, y_time_stretched, y_noisy]
```

**Audio Processing Details:**
- **Sample Rate**: 22050 Hz
- **Original Duration**: 2.76s - 5.74s (Christophe: 2.76s, Anne: 5.74s)
- **Trimmed Duration**: 1.49s - 3.04s (after silence removal)
- **Processing**: Top_db=30 for silence trimming

**Feature Extraction Development:**
- Implemented **MFCC** feature extraction (20 coefficients)
- Created spectral feature extraction (roll-off, RMS energy)
- Developed 44-dimensional feature vectors
- Generated `audio_features.csv` with 32 samples (8 originals × 4 augmentations)

**Voiceprint Verification Model Training:**
- Implemented Random Forest classifier for voice verification
- Achieved 85.71% accuracy on test set
- Created comprehensive model evaluation
- Developed confidence-based verification system

**Files Created/Modified:**
- `Data/audios/` - Original audio collection
- `Datasets/audio_features.csv` - Extracted audio features
- `models/voiceprint_verification_model.joblib` - Trained model
- `encoders/voice_feature_scaler.joblib` - Feature scaler
- `Notebooks/Audio_Processing_Features.ipynb` - Processing notebook

---

### 4. Kanisa Rebecca Majok Thiak - System Demo Implementation and Simulation

#### Task 4: System Demo and Integration

**Demo System Architecture:**
- Developed `BiometricSecuritySystem` class
- Implemented interactive menu system for the demo simulation
- Created real-time authentication workflow
- Developed unauthorized access simulation

**System Integration Implementation:**
```python
class BiometricSecuritySystem:
    def __init__(self):
        # Load all models and scalers
        self.load_models()
        self.load_customer_data()
        
    def run_full_transaction(self, user_name, image_path, audio_path):
        # Step 1: Face Authentication
        face_auth_success, predicted_user = self.authenticate_face(image_path)
        if not face_auth_success:
            return False
        
        # Step 2: Voice Verification
        voice_auth_success, final_user = self.verify_voice(audio_path, predicted_user)
        if not voice_auth_success:
            return False
        
        # Step 3: Product Recommendation
        recommended_category, user_profile = self.get_product_recommendations(final_user)
        
        # Step 4: Transaction Approval
        print("All authentication steps passed successfully!")
        print("User authorized to proceed with transaction")
        print(f"Recommended products: {recommended_category}")
        print("TRANSACTION COMPLETED SUCCESSFULLY!")
        
        return True
```

**Interactive Demo Features:**
- **Authorized User Simulation**: Complete transaction flow for all team members
- **Unauthorized Access Simulation**: Security denial demonstration
- **Custom Transaction**: User-defined image and audio paths
- **Real-time Authentication**: Live face and voice verification

**Demo Menu System:**
```python
def main_menu(self):
    print("1.  Simulate Authorized Transaction (Anne)")
    print("2.  Simulate Authorized Transaction (Christophe)")
    print("3.  Simulate Authorized Transaction (Excel)")
    print("4.  Simulate Authorized Transaction (Kanisa)")
    print("5.  Simulate Unauthorized Attempt")
    print("6.  Custom Transaction (Specify paths)")
    print("7.  Exit")
```

**Setup and Environment Management:**
- Created `setup_demo.py` for automated environment setup
- Implemented dependency management with `requirements.txt`
- Developed file validation and error handling
- Saw that installation procedures are handled

**Files Created/Modified:**
- `system_demo.py` - Main demo application
- `setup_demo.py` - Environment setup script
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

---

## GitHub Repository Information

**Repository**: [https://github.com/excelasaph/Data-Preprocessing-Group-11](https://github.com/excelasaph/Data-Preprocessing-Group-11)  
**Main Branch**: main  
**Contributors**: 4 team members  
**Total Commits**: 30+ commits  
**Languages**: Python, Jupyter Notebook, Markdown  
**Technologies**: OpenCV, Librosa, Scikit-learn, XGBoost, Joblib