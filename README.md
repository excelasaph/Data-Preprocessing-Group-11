# User Identity and Product Recommendation System

A multimodal biometric security system that implements facial recognition, voiceprint verification, and personalized product recommendations using machine learning techniques.

## Project Overview

This project implements a secure, multimodal user authentication and product recommendation system. The system uses facial recognition and voiceprint verification to authenticate users before providing personalized product recommendations. The workflow ensures that only verified users can access the recommendation model, with clear denial pathways for failed authentication.

### Key Features

- **Multi-modal Authentication**: Sequential face and voice verification
- **Facial Recognition**: XGBoost-based face recognition using color histogram features
- **Voiceprint Verification**: Random Forest-based voice verification using MFCC features
- **Product Recommendation**: Personalized product category predictions
- **Data Augmentation**: Comprehensive image and audio augmentation pipeline
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
│              Authentication & Recommendation Engine              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Face      │  │   Voice     │  │   Product               │ │
│  │   Auth      │  │   Verify    │  │   Recommend             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
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
   git clone <repository-url>
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
📹 **[Watch the System Demo on YouTube](https://youtube.com/watch?v=YOUR_VIDEO_ID)**

The demo showcases:
- Real-time facial recognition
- Voiceprint verification
- Personalized product recommendations
- Unauthorized access simulation
- Interactive menu system

### Demo Features

#### Authorized User Simulation
- **Face Authentication**: Upload team member photos (neutral, smiling, surprised)
- **Voice Verification**: Process audio samples ("Yes, approve", "Confirm transaction")
- **Product Recommendation**: Get personalized product category predictions
- **Transaction Approval**: Complete secure transaction flow

#### Unauthorized Access Simulation
- **Access Denial**: Demonstrate security measures
- **Alert System**: Show security logging and alerts
- **Incident Tracking**: Log unauthorized attempts with timestamps

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

## Project Structure

```
Data-Preprocessing-Group-11/
├── Data/                          # Raw data collection
│   ├── audios/                    # Original audio samples
│   ├── pictures/                  # Original images
│   │   ├── neutral/              # Neutral expressions
│   │   ├── smilling/             # Smiling expressions
│   │   └── surprised/            # Surprised expressions
│   └── datasets/                  # Customer data
│       ├── customer_social_profiles.csv
│       └── customer_transactions.csv
├── Datasets/                      # Processed datasets
│   ├── image_features.csv         # Extracted image features
│   ├── audio_features.csv         # Extracted audio features
│   └── merged_customer_data.csv   # Merged customer data
├── models/                        # Trained models
│   ├── facial_recognition_xgboost_model.joblib
│   ├── voiceprint_verification_model.joblib
│   └── product_recommendation_model.pkl
├── encoders/                      # Feature scalers
│   ├── voice_feature_scaler.joblib
│   ├── product_recommendation_scaler.pkl
│   └── facial_recognition_label_encoder.joblib
├── Notebooks/                     # Jupyter notebooks
│   ├── Audio_Processing_Features.ipynb
│   ├── Data_Merging_Product_Recommendation_Model_Training.ipynb
│   └── Image_processing&_Facial_recognition_model.ipynb
├── augmented/                     # Augmented images
├── system_demo.py                 # Main demo application
├── setup_demo.py                  # Environment setup
├── requirements.txt               # Python dependencies
└── README.md                     # Project documentation
```

## Technical Details

### Authentication Flow
1. **Face Recognition**: Extract color histogram features → XGBoost prediction
2. **Voice Verification**: Extract MFCC features → Random Forest prediction
3. **Multi-factor Validation**: Both modalities must pass for access
4. **Product Recommendation**: Customer profile analysis → category prediction

### Security Features
- **Confidence Thresholds**: Minimum confidence levels for authentication
- **Multi-modal Validation**: Sequential face + voice verification
- **Unauthorized Detection**: Clear denial pathways for failed authentication
- **Incident Logging**: Security alerts and timestamp tracking

### Data Processing Features
- **Comprehensive Augmentation**: 4 versions per sample (original + 3 augmentations)
- **Feature Engineering**: Automated feature extraction pipeline
- **Model Persistence**: Saved models and scalers for production use
- **Error Handling**: Robust error handling and validation

## Model Performance

### Facial Recognition
- **Model**: XGBoost with color histogram features
- **Classes**: 3 (mapped to team members)
- **Features**: 512-dimensional color histogram
- **Augmentation**: 4 versions per image

### Voice Verification
- **Model**: Random Forest with MFCC features
- **Accuracy**: 85.71% on test set
- **Classes**: 4 (one per team member)
- **Features**: 44-dimensional feature vector

### Product Recommendation
- **Model**: Random Forest with hyperparameter tuning
- **Accuracy**: 65% on test set
- **Features**: 15 engineered customer features
- **Categories**: 5 product categories

## Assignment Requirements Coverage

✅ **Data Merge**: Customer profiles + transactions merged into `merged_customer_data.csv`

✅ **Image Data Collection**: 
- 4 team members × 3 expressions = 12 original images
- Comprehensive augmentation (rotation, flipping, grayscale)
- Feature extraction and storage in `image_features.csv`

✅ **Audio Data Collection**:
- 4 team members × 2 phrases = 8 original audio samples
- Multiple augmentations (pitch shift, time stretch, noise)
- Feature extraction and storage in `audio_features.csv`

✅ **Model Creation**:
- **Facial Recognition Model**: XGBoost classifier
- **Voiceprint Verification Model**: Random Forest classifier
- **Product Recommendation Model**: Random Forest with hyperparameter tuning

✅ **System Demonstration**:
- Interactive demo with authorized user simulation
- Unauthorized access simulation
- Real-time authentication workflow
- Product recommendation integration

✅ **Evaluation Metrics**:
- **Accuracy**: Face (optimized), Voice (85.71%), Product (65%)
- **F1-Score**: Comprehensive classification reports
- **Loss**: Model performance tracking

## Usage Examples

### Running the Demo
```bash
# Install dependencies
python setup_demo.py

# Start the demo system
python system_demo.py
```

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

## Deployment

### Local Development
```bash
# Set up environment
python setup_demo.py

# Run demo system
python system_demo.py
```

### Production Considerations
- **Model Optimization**: Further hyperparameter tuning
- **Security Enhancement**: Additional authentication factors
- **Scalability**: Cloud deployment with load balancing
- **Monitoring**: Real-time performance monitoring

## Team Members

**Data Preprocessing Group 11**:
- **Anne Marie Twagirayezu** - Image processing and facial recognition
- **Excel Asaph** - Data merging and product recommendation
- **Christophe Gakwaya** - Audio processing and voice verification
- **Kanisa Rebecca Majok Thiak** - System architecture and demo development

## License

This project is developed for educational purposes as part of the Data Preprocessing course assignment.

## Acknowledgments

- **Course Instructors**: For guidance on multimodal data preprocessing
- **Team Collaboration**: Successful implementation of complex biometric system
- **Open Source Libraries**: OpenCV, Librosa, Scikit-learn, XGBoost