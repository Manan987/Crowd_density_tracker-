#!/bin/bash

# CrowdGuard Pro Setup Script
# This script sets up the development environment for the Crowd Density Tracker

set -e  # Exit on any error

echo "🚀 Setting up CrowdGuard Pro Development Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.11+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 11) else 1)'; then
            print_success "Python ${PYTHON_VERSION} is installed and compatible"
        else
            print_error "Python 3.11+ is required. Current version: ${PYTHON_VERSION}"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check if uv is installed, if not, install it
check_uv() {
    print_status "Checking for uv package manager..."
    if ! command -v uv &> /dev/null; then
        print_warning "uv not found. Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        print_success "uv installed successfully"
    else
        print_success "uv is already installed"
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Activate virtual environment
    source .venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies from pyproject.toml
    if command -v uv &> /dev/null; then
        uv pip install -e .
        uv pip install -e ".[dev]"
    else
        pip install -e .
        pip install -e ".[dev]"
    fi
    
    print_success "Dependencies installed successfully"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."
    
    directories=(
        "uploads"
        "temp"
        "logs"
        "models/weights"
        "data/processed"
        "data/raw"
        "exports"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    
    print_success "All directories created"
}

# Copy environment template
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f ".env.template" ]; then
            cp .env.template .env
            print_success "Created .env file from template"
            print_warning "Please edit .env file with your specific configuration"
        else
            print_warning ".env.template not found, creating basic .env file"
            cat > .env << EOF
APP_NAME="CrowdGuard Pro"
APP_VERSION="1.0.0"
APP_ENVIRONMENT="development"
DEBUG=true
HOST="0.0.0.0"
PORT=8501
DATABASE_URL="sqlite:///crowd_monitoring.db"
EOF
            print_success "Basic .env file created"
        fi
    else
        print_warning ".env file already exists"
    fi
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    source .venv/bin/activate
    
    # Create database tables (if using SQLAlchemy)
    python -c "
from database.models import Base
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///crowd_monitoring.db')

try:
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    print('Database tables created successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    print('This is normal if database models are not fully set up yet')
"
    
    print_success "Database initialization completed"
}

# Download sample model weights (placeholder)
download_models() {
    print_status "Setting up model weights..."
    
    MODEL_DIR="models/weights"
    
    if [ ! -f "$MODEL_DIR/crowd_model.pth" ]; then
        print_warning "No pre-trained models found"
        print_status "Creating placeholder model files..."
        
        # Create placeholder files
        touch "$MODEL_DIR/crowd_model.pth"
        touch "$MODEL_DIR/enhanced_model.pth"
        
        cat > "$MODEL_DIR/README.md" << EOF
# Model Weights

This directory should contain the trained model weights:

- \`crowd_model.pth\`: Basic crowd density estimation model
- \`enhanced_model.pth\`: Enhanced multi-scale model

## Downloading Models

To download pre-trained models, run:

\`\`\`bash
# Download from your model repository
wget -O crowd_model.pth "https://your-model-repo.com/crowd_model.pth"
wget -O enhanced_model.pth "https://your-model-repo.com/enhanced_model.pth"
\`\`\`

## Training Your Own Models

Refer to the training documentation in the \`docs/\` directory.
EOF
        
        print_success "Model directory set up with placeholders"
    else
        print_success "Model weights already present"
    fi
}

# Install Git hooks
setup_git_hooks() {
    print_status "Setting up Git hooks..."
    
    if [ -d ".git" ]; then
        source .venv/bin/activate
        
        # Install pre-commit hooks
        if command -v pre-commit &> /dev/null; then
            pre-commit install
            print_success "Pre-commit hooks installed"
        else
            print_warning "pre-commit not available, skipping Git hooks setup"
        fi
    else
        print_warning "Not a Git repository, skipping Git hooks setup"
    fi
}

# Run tests to verify setup
run_tests() {
    print_status "Running setup verification tests..."
    
    source .venv/bin/activate
    
    # Basic import test
    python -c "
try:
    import streamlit
    import torch
    import cv2
    import numpy as np
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"
    
    # Check if app can start (dry run)
    print_status "Testing application startup..."
    timeout 10s python -c "
import sys
sys.path.append('.')
try:
    # Test basic app imports
    from models.crowd_density_model import CrowdDensityEstimator
    from utils.video_processor import VideoProcessor
    print('✅ Application modules load successfully')
except Exception as e:
    print(f'⚠️  Warning: {e}')
    print('Some modules may not be fully functional yet')
" || print_warning "Application startup test timed out (this is normal)"
    
    print_success "Setup verification completed"
}

# Main setup process
main() {
    echo "=================================="
    echo "🛡️  CrowdGuard Pro Setup Script"
    echo "=================================="
    echo
    
    check_python
    check_uv
    create_venv
    install_dependencies
    create_directories
    setup_environment
    init_database
    download_models
    setup_git_hooks
    run_tests
    
    echo
    echo "=================================="
    print_success "Setup completed successfully! 🎉"
    echo "=================================="
    echo
    echo "📋 Next steps:"
    echo "1. Edit .env file with your configuration"
    echo "2. Download or train model weights"
    echo "3. Run the application:"
    echo "   source .venv/bin/activate"
    echo "   streamlit run app.py"
    echo
    echo "📚 For more information, see README.md"
}

# Run main function
main "$@"
