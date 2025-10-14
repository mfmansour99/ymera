#!/bin/bash

# YMERA Enterprise Platform - Quick Start Script
# This script helps you get started with YMERA quickly

set -e  # Exit on error

echo "=========================================="
echo "YMERA Enterprise Platform - Quick Start"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Check Python version
echo "Checking prerequisites..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_status "Python installed: $PYTHON_VERSION"
else
    print_error "Python 3.9+ is required but not installed"
    exit 1
fi

# Check Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    print_status "Node.js installed: $NODE_VERSION"
else
    print_warning "Node.js not found. Frontend development will not be available."
fi

# Check Docker
if command -v docker &> /dev/null; then
    print_status "Docker is installed"
    DOCKER_AVAILABLE=true
else
    print_warning "Docker not found. Some features may not be available."
    DOCKER_AVAILABLE=false
fi

echo ""
echo "=========================================="
echo "Setup Options:"
echo "=========================================="
echo "1. Quick Start (Docker - Recommended)"
echo "2. Development Setup (Manual)"
echo "3. Production Deployment"
echo "4. Exit"
echo ""
read -p "Choose an option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "=== Quick Start with Docker ==="
        if [ "$DOCKER_AVAILABLE" = false ]; then
            print_error "Docker is not available. Please install Docker first."
            exit 1
        fi
        
        # Check if .env exists
        if [ ! -f .env ]; then
            print_warning ".env file not found. Creating from template..."
            cp .env.example .env
            print_status ".env file created. Please edit it with your configuration."
            print_warning "Press Enter after editing .env to continue..."
            read
        fi
        
        echo "Building and starting containers..."
        docker-compose up -d
        
        print_status "YMERA is starting..."
        echo ""
        echo "Waiting for services to be ready..."
        sleep 10
        
        echo ""
        echo "=========================================="
        echo "✓ YMERA is running!"
        echo "=========================================="
        echo "Access points:"
        echo "  - API: http://localhost:8000"
        echo "  - Docs: http://localhost:8000/docs"
        echo "  - Health: http://localhost:8000/health"
        echo ""
        echo "To view logs: docker-compose logs -f"
        echo "To stop: docker-compose down"
        ;;
        
    2)
        echo ""
        echo "=== Development Setup ==="
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            print_status "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        print_status "Activating virtual environment..."
        source venv/bin/activate
        
        # Install Python dependencies
        print_status "Installing Python dependencies..."
        pip install -q -r requirements.txt
        
        # Check if .env exists
        if [ ! -f .env ]; then
            print_warning ".env file not found. Creating from template..."
            cp .env.example .env
            print_status ".env file created. Please edit it before running."
        fi
        
        # Install Node dependencies if Node is available
        if command -v npm &> /dev/null; then
            print_status "Installing Node.js dependencies..."
            npm install --silent
        fi
        
        echo ""
        echo "=========================================="
        echo "✓ Development environment ready!"
        echo "=========================================="
        echo "To start the backend:"
        echo "  source venv/bin/activate"
        echo "  python main_production.py"
        echo ""
        echo "To start the frontend:"
        echo "  npm run dev"
        echo ""
        echo "Don't forget to configure your .env file!"
        ;;
        
    3)
        echo ""
        echo "=== Production Deployment ==="
        echo ""
        echo "For production deployment, please refer to DEPLOYMENT.md"
        echo "Key steps:"
        echo "1. Configure .env with production values"
        echo "2. Set up PostgreSQL and Redis"
        echo "3. Run database migrations"
        echo "4. Deploy with Docker or manual setup"
        echo ""
        echo "See DEPLOYMENT.md for detailed instructions."
        ;;
        
    4)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        print_error "Invalid option"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Need help? Check these resources:"
echo "  - README.md - Overview and documentation"
echo "  - DEPLOYMENT.md - Deployment guide"
echo "  - CONTRIBUTING.md - Development guide"
echo "  - SECURITY.md - Security best practices"
echo "=========================================="
