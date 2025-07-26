#!/usr/bin/env python3
"""
Setup Script for Biometric Security System Demo
===============================================
This script helps set up the environment for running the demo.

Usage:
    python setup_demo.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_files():
    """Check if all required files exist."""
    print("🔍 Checking for required files...")
    
    required_files = [
        "models/facial_recognition_xgboost.joblib",
        "models/voiceprint_verification_model.joblib", 
        "models/product_recommendation_model.pkl",
        "encoders/voice_feature_scaler.joblib",
        "encoders/product_recommendation_scaler.pkl",
        "Datasets/merged_customer_data.csv",
        "Data/pictures/neutral/anne.jpg",
        "Data/pictures/neutral/christophe.jpg",
        "Data/pictures/neutral/excel.jpg",
        "Data/pictures/neutral/kanisa.jpg",
        "Data/audios/Anne_confirm_transaction.wav",
        "Data/audios/Christophe_confirm_transaction.wav",
        "Data/audios/Excel_confirm_transaction.wav",
        "Data/audios/kanisa_confirm_transaction.wav"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n⚠️  Please ensure all models are trained and data files are present.")
        return False
    else:
        print("✅ All required files found!")
        return True

def main():
    """Main setup function."""
    print("🚀 Setting up Biometric Security System Demo")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation.")
        return False
    
    # Check files
    if not check_files():
        print("❌ Setup failed - missing required files.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("🔐 You can now run the demo with: python system_demo.py")
    return True

if __name__ == "__main__":
    main()
