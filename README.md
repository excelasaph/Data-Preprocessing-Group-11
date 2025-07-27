# User Identity and Product Recommendation System

A multimodal biometric security system that implements facial recognition, voiceprint verification, and personalized product recommendations using machine learning techniques.

## Project Overview

This project implements a secure, multimodal user authentication and product recommendation system. The system uses facial recognition and voiceprint verification to authenticate users before providing personalized product recommendations. The workflow ensures that only verified users can access the recommendation model, with clear denial pathways for failed authentication.

## Project Structure

```
Data-Preprocessing-Group-11/
├── Data/                          
│   ├── audios/                   
│   ├── pictures/                  
│   │   ├── neutral/              
│   │   ├── smilling/            
│   │   └── surprised/            
│   └── datasets/                  
│       ├── customer_social_profiles.csv
│       └── customer_transactions.csv
├── Datasets/                      
│   ├── image_features.csv         
│   ├── audio_features.csv         
│   └── merged_customer_data.csv   
├── models/                        
│   ├── facial_recognition_xgboost_model.joblib
│   ├── voiceprint_verification_model.joblib
│   └── product_recommendation_model.pkl
├── encoders/                      
│   ├── voice_feature_scaler.joblib
│   ├── product_recommendation_scaler.pkl
│   └── facial_recognition_label_encoder.joblib
├── Notebooks/                     
│   ├── Audio_Processing_Features.ipynb
│   ├── Data_Merging_Product_Recommendation_Model_Training.ipynb
│   └── Image_processing&_Facial_recognition_model.ipynb
├── report/                        
│   ├── report.md                  
│   └── report.pdf                 
├── augmented/                     
├── system_demo.py                 
├── setup_demo.py                  
├── requirements.txt               
└── README.md                     
```

### Key Features

- **Multi-modal Authentication**: Sequential face and voice verification
- **Facial Recognition**: XGBoost-based face recognition using color histogram features
- **Voiceprint Verification**: Random Forest-based voice verification using MFCC features
- **Product Recommendation**: Personalized product category predictions
- **Data Augmentation**: Image and audio augmentation pipeline
- **Interactive Demo**: Real-time system demonstration with unauthorized access simulation
- **Security Features**: Multi-factor authentication with clear denial pathways

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Face Image    │    │   Voice Audio   │    │  Customer Data  │
│   Processing    │    │   Processing    │    │   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Face Model     │    │  Voice Model    │    │ Product Model   │
│  (XGBoost)      │    │ (Random Forest) │    │ (Random Forest) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│              Authentication & Recommendation Engine             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Face      │  │   Voice     │  │   Product               │  │
│  │   Auth      │  │   Verify    │  │   Recommend             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Python 3.8+
- OpenCV for computer vision
- Librosa for audio processing
- Scikit-learn for machine learning

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/excelasaph/Data-Preprocessing-Group-11.git
   cd Data-Preprocessing-Group-11
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run setup script**
   ```bash
   python setup_demo.py
   ```

4. **Start the demo system**
   ```bash
   python system_demo.py  
   ```

## System Demo

### Demo Video
**[Watch the System Demo on YouTube](https://youtube.com/watch?v=YOUR_VIDEO_ID)**

### Demo Menu Options
1. **Simulate Authorized Transaction (Anne)**
2. **Simulate Authorized Transaction (Christophe)**
3. **Simulate Authorized Transaction (Excel)**
4. **Simulate Authorized Transaction (Kanisa)**
5. **Simulate Unauthorized Attempt**
6. **Custom Transaction (Specify paths)**
7. **Exit**

### Example Transaction Flow
```
STEP 1: FACE AUTHENTICATION
- Analyzing face image: anne.jpg
- Predicted identity: anne
- Confidence: 78.5%
- Face authentication SUCCESSFUL for anne

STEP 2: VOICE VERIFICATION
- Analyzing voice sample: Anne_confirm_transaction.wav
- Predicted voice identity: anne
- Voice confidence: 82.3%
- Voice verification SUCCESSFUL

STEP 3: PRODUCT RECOMMENDATION SYSTEM
- Generating product recommendations for user: anne
- Recommended Category: Books

STEP 4: TRANSACTION APPROVAL
- All authentication steps passed successfully!
- User authorized to proceed with transaction
- Recommended products: Books
- TRANSACTION COMPLETED SUCCESSFULLY!
```

## Dataset Information

### Image Dataset
- **Source**: Team member photos
- **Format**: JPG images
- **Expressions**: Neutral, Smiling, Surprised
- **Augmentation**: 4 versions per image

### Audio Dataset
- **Source**: Team member voice recordings
- **Format**: WAV files
- **Phrases**: "Yes, approve", "Confirm transaction"
- **Augmentation**: 4 versions per audio

### Customer Dataset
- **Social Profiles**: Engagement scores, social media platforms
- **Transaction History**: Purchase amounts, frequencies, ratings
- **Features**: 15 engineered features for recommendation

## Data Processing Pipeline

### Image Data Collection and Processing

#### Data Collection
- **Team Members**: Anne, Christophe, Excel, Kanisa
- **Expressions**: Neutral, Smiling, Surprised (3 per member)
- **Total Images**: 12 original images

#### Augmentation Pipeline
- **Rotation**: ±15 degrees
- **Flipping**: Horizontal and vertical
- **Grayscale**: Color to grayscale conversion
- **Total Augmented**: 48 images (12 × 4 augmentations)

#### Feature Extraction
- **Method**: Color histogram features (RGB channels)
- **Features**: 512-dimensional feature vector (8×8×8 bins)
- **Output**: `image_features.csv`

### Audio Data Collection and Processing

#### Data Collection
- **Team Members**: Anne, Christophe, Excel, Kanisa
- **Phrases**: "Yes, approve", "Confirm transaction"
- **Total Audio**: 8 original samples

#### Augmentation Pipeline
- **Pitch Shift**: ±2 semitones
- **Time Stretch**: 1.2x speed variation
- **Noise Addition**: 0.5% amplitude noise
- **Total Augmented**: 32 samples (8 × 4 augmentations)

#### Feature Extraction
- **MFCCs**: 20 Mel-frequency cepstral coefficients
- **Spectral Features**: Roll-off frequency, RMS energy
- **Total Features**: 44-dimensional feature vector
- **Output**: `audio_features.csv`

### Data Merging and Product Recommendation

#### Customer Data Integration
- **Social Profiles**: Customer engagement and social media metrics
- **Transaction History**: Purchase patterns and preferences
- **Merged Dataset**: 84 customer records with 15 engineered features

#### Feature Engineering
- **Engagement Metrics**: Social media activity scores
- **Purchase Patterns**: Transaction frequency and amounts
- **Temporal Features**: Purchase date encoding
- **Categorical Encoding**: One-hot encoding for platforms and sentiments
- **Output**: `merged_customer_data.csv`

## Machine Learning Models

### Facial Recognition Model
- **Algorithm**: XGBoost Classifier
- **Features**: Color histogram (512 dimensions)
- **Classes**: 3 (mapped to team members)
- **Performance**: Optimized for demo accuracy
- **Output**: `facial_recognition_xgboost_model.joblib`

### Voiceprint Verification Model
- **Algorithm**: Random Forest Classifier
- **Features**: MFCC + Spectral features (44 dimensions)
- **Classes**: 4 (one per team member)
- **Accuracy**: 85.71% on test set
- **Output**: `voiceprint_verification_model.joblib`

### Product Recommendation Model
- **Algorithm**: Random Forest with hyperparameter tuning
- **Features**: 15 engineered customer features
- **Categories**: Electronics, Clothing, Books, Sports, Groceries
- **Accuracy**: 65% on test set
- **Output**: `product_recommendation_model.pkl`

## Team Members

**Data Preprocessing Group 11**:
- **Anne Marie Twagirayezu** - Image processing and facial recognition
- **Excel Asaph** - Data merging and product recommendation
- **Christophe Gakwaya** - Audio processing and voice verification
- **Kanisa Rebecca Majok Thiak** - System architecture and demo development

## Acknowledgments

- **Open Source Libraries**: OpenCV, Librosa, Scikit-learn, XGBoost