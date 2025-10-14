# YMERA Enterprise Platform - Deployment Guide

## 📋 Pre-Deployment Checklist

Before deploying YMERA to production, ensure you have completed the following:

### 1. Environment Setup

- [ ] Python 3.9+ installed
- [ ] Node.js 18+ installed
- [ ] PostgreSQL 13+ running
- [ ] Redis 6+ running
- [ ] All environment variables configured in `.env`

### 2. Security Configuration

- [ ] Changed all default passwords and secret keys
- [ ] Configured CORS origins for your domain
- [ ] Set up SSL/TLS certificates
- [ ] Reviewed and updated SECURITY.md
- [ ] Enabled rate limiting
- [ ] Configured firewall rules

### 3. Database Setup

- [ ] Created production database
- [ ] Configured database connection string
- [ ] Ran database migrations
- [ ] Set up database backups
- [ ] Configured connection pooling

### 4. Application Configuration

- [ ] Updated `.env` with production values
- [ ] Configured logging levels
- [ ] Set up error tracking (Sentry, etc.)
- [ ] Configured monitoring
- [ ] Set up health check endpoints

## 🚀 Deployment Methods

### Method 1: Docker Deployment (Recommended)

#### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mfmansour99/ymera.git
cd ymera

# 2. Configure environment
cp .env.example .env
# Edit .env with your production values

# 3. Build and start containers
docker-compose up -d

# 4. Check logs
docker-compose logs -f

# 5. Access the application
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

#### Production Docker Setup

```bash
# Build production image
docker build -t ymera:production .

# Run with environment file
docker run -d \
  --name ymera \
  -p 8000:8000 \
  --env-file .env \
  ymera:production

# Check health
curl http://localhost:8000/health
```

### Method 2: Manual Deployment

#### Backend Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up database
export DATABASE_URL="postgresql://user:pass@localhost/ymera"
python -c "from app.database.database import create_tables; create_tables()"

# 4. Run migrations (if using Alembic)
alembic upgrade head

# 5. Start the server
python main_production.py

# Or with uvicorn for production:
uvicorn main_production:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Frontend Setup

```bash
# 1. Install dependencies
npm install

# 2. Build for production
npm run build

# 3. The built files will be in ./dist
# Serve them with your web server (nginx, Apache, etc.)
```

### Method 3: Cloud Platform Deployment

#### Heroku

```bash
# 1. Create Heroku app
heroku create ymera-production

# 2. Add PostgreSQL
heroku addons:create heroku-postgresql:hobby-dev

# 3. Add Redis
heroku addons:create heroku-redis:hobby-dev

# 4. Set environment variables
heroku config:set SECRET_KEY=your-secret-key
heroku config:set ENVIRONMENT=production

# 5. Deploy
git push heroku main
```

#### AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 22.04)
# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install python3.11 python3-pip nginx postgresql redis-server

# 4. Clone and setup
git clone https://github.com/mfmansour99/ymera.git
cd ymera
pip install -r requirements.txt

# 5. Configure nginx as reverse proxy
sudo nano /etc/nginx/sites-available/ymera

# 6. Start with systemd
sudo systemctl start ymera
```

## 🔧 Configuration

### Essential Environment Variables

```bash
# Application
APP_NAME=YMERA Enterprise
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-super-secure-secret-key-change-this

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ymera
DB_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# AI Services (optional)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

## 🔍 Health Checks

### Application Health

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed metrics
curl http://localhost:8000/metrics

# API documentation
curl http://localhost:8000/docs
```

### Database Health

```bash
# Check PostgreSQL
psql $DATABASE_URL -c "SELECT 1;"

# Check Redis
redis-cli ping
```

## 📊 Monitoring

### Logging

Logs are written to:
- **Console**: Standard output/error
- **File**: `./logs/app.log` (if configured)
- **Remote**: Configured logging service (Sentry, etc.)

### Metrics

Monitor these key metrics:
- Response time
- Error rate
- CPU/Memory usage
- Database connection pool
- Queue length
- Active connections

## 🔄 Updates and Maintenance

### Rolling Updates

```bash
# 1. Pull latest changes
git pull origin main

# 2. Backup database
pg_dump $DATABASE_URL > backup.sql

# 3. Update dependencies
pip install -r requirements.txt
npm install

# 4. Run migrations
alembic upgrade head

# 5. Restart application
docker-compose restart app
# OR
systemctl restart ymera
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Solution: Ensure app directory is in Python path
export PYTHONPATH=$PYTHONPATH:/path/to/ymera
```

**Database Connection Failed**
```bash
# Check connection string
echo $DATABASE_URL

# Test connection
psql $DATABASE_URL

# Check PostgreSQL is running
systemctl status postgresql
```

**Port Already in Use**
```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>
```

## 📞 Support

For deployment issues:
1. Check logs: `docker-compose logs -f` or `journalctl -u ymera`
2. Review [GitHub Issues](https://github.com/mfmansour99/ymera/issues)
3. Contact maintainers

## ✅ Post-Deployment Checklist

- [ ] Application is accessible
- [ ] Health check returns healthy status
- [ ] Authentication works
- [ ] Database queries are successful
- [ ] File uploads work
- [ ] WebSocket connections work
- [ ] Monitoring is active
- [ ] Backups are configured
- [ ] SSL certificate is valid
- [ ] Performance is acceptable
- [ ] Error tracking is working
- [ ] Documentation is updated

## 🔐 Security Hardening

After deployment:

1. **Firewall Configuration**
   - Only expose necessary ports (80, 443)
   - Block direct database access from internet
   - Configure rate limiting

2. **SSL/TLS**
   - Use Let's Encrypt for free SSL
   - Force HTTPS redirect
   - Enable HSTS headers

3. **Regular Updates**
   - Keep dependencies updated
   - Apply security patches promptly
   - Monitor security advisories

4. **Access Control**
   - Use SSH keys (no passwords)
   - Implement 2FA for admin accounts
   - Regular security audits

---

**Need help?** Open an issue on GitHub or contact the maintainers.
