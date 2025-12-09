# 🚀 Starting the Cloud Intelligence Platform

## Step-by-Step Startup Guide

### ⚠️ Prerequisites Check

Before starting, ensure you have:
- ✅ Docker Desktop installed
- ✅ Docker Desktop is **RUNNING** (check system tray)
- ✅ At least 8GB RAM available
- ✅ At least 20GB disk space

---

## 🎯 Quick Start (3 Steps)

### Step 1: Start Docker Desktop

**Windows:**
1. Open Start Menu
2. Search for "Docker Desktop"
3. Click to launch
4. Wait for Docker to start (whale icon in system tray)
5. Icon should be steady (not animated)

**Verify Docker is running:**
```powershell
docker ps
```
Should show: `CONTAINER ID   IMAGE   ...` (even if empty)

---

### Step 2: Start the Platform

**Option A: Using PowerShell Script (Recommended)**
```powershell
.\start-project.ps1
```

**Option B: Using Docker Compose**
```powershell
docker-compose up -d
```

**Option C: Using Python Script**
```powershell
python start-dev.py
```

---

### Step 3: Wait for Services to Start

This takes **2-5 minutes** on first run (downloading images).

**Check status:**
```powershell
docker-compose ps
```

**Watch logs:**
```powershell
docker-compose logs -f
```

Press `Ctrl+C` to stop watching logs (services keep running).

---

## 🌐 Access the Platform

Once all services are running:

### Main Application
- **Frontend (Web UI):** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

### Monitoring & Logs
- **Grafana (Metrics):** http://localhost:3001
  - Username: `admin`
  - Password: `admin`
- **Prometheus:** http://localhost:9090
- **Kibana (Logs):** http://localhost:5601

### Databases
- **PostgreSQL:** localhost:5432
  - Database: `finops_db`
  - Username: `finops`
  - Password: `finops_password`
- **Redis:** localhost:6379
  - Password: `redis_password`

---

## ✅ Verify Everything is Working

### 1. Check All Services are Running
```powershell
docker-compose ps
```

You should see 15 services with status "Up":
- postgres
- redis
- api
- frontend
- worker
- scheduler
- prometheus
- grafana
- node-exporter
- cadvisor
- elasticsearch
- logstash
- kibana
- filebeat

### 2. Test the Frontend
Open browser: http://localhost:3000

You should see the **Cloud Intelligence Platform** landing page.

### 3. Test the Backend API
Open browser: http://localhost:8000/docs

You should see the **Swagger API documentation**.

### 4. Check Health Endpoints
```powershell
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000
```

---

## 🐛 Troubleshooting

### Problem: Docker Desktop Not Running
**Error:** `The system cannot find the file specified`

**Solution:**
1. Start Docker Desktop from Start Menu
2. Wait for it to fully start (30-60 seconds)
3. Try again

---

### Problem: Port Already in Use
**Error:** `port is already allocated`

**Solution:**
```powershell
# Find what's using the port (e.g., 3000)
netstat -ano | findstr :3000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F

# Or change the port in docker-compose.override.yml
```

---

### Problem: Services Not Starting
**Error:** Container exits immediately

**Solution:**
```powershell
# Check logs for specific service
docker-compose logs api
docker-compose logs postgres

# Restart specific service
docker-compose restart api

# Rebuild and restart
docker-compose up -d --build
```

---

### Problem: Out of Memory
**Error:** Container killed or OOM

**Solution:**
1. Close other applications
2. Increase Docker Desktop memory:
   - Settings → Resources → Memory
   - Set to at least 8GB
3. Restart Docker Desktop

---

### Problem: Database Connection Failed
**Error:** `could not connect to server`

**Solution:**
```powershell
# Wait for postgres to be ready (takes 30-60 seconds)
docker-compose logs postgres

# Look for: "database system is ready to accept connections"

# Restart API after postgres is ready
docker-compose restart api
```

---

## 🛑 Stopping the Platform

### Stop All Services (Keep Data)
```powershell
docker-compose stop
```

### Stop and Remove Containers (Keep Data)
```powershell
docker-compose down
```

### Stop and Remove Everything (Including Data)
```powershell
docker-compose down -v
```
⚠️ **Warning:** This deletes all data!

---

## 🔄 Restarting Services

### Restart All Services
```powershell
docker-compose restart
```

### Restart Specific Service
```powershell
docker-compose restart api
docker-compose restart frontend
```

### Rebuild and Restart (After Code Changes)
```powershell
docker-compose up -d --build
```

---

## 📊 Monitoring Startup Progress

### Watch All Logs
```powershell
docker-compose logs -f
```

### Watch Specific Service
```powershell
docker-compose logs -f api
docker-compose logs -f postgres
```

### Check Service Health
```powershell
docker-compose ps
```

Look for "healthy" status in the STATE column.

---

## 🎯 First-Time Setup Checklist

After starting the platform for the first time:

- [ ] All 15 services are running
- [ ] Frontend loads at http://localhost:3000
- [ ] API docs load at http://localhost:8000/docs
- [ ] Can create a user account
- [ ] Can login successfully
- [ ] Dashboard loads without errors

---

## 🚀 Next Steps

Once the platform is running:

1. **Create an Account**
   - Go to http://localhost:3000
   - Click "Sign Up" or use API at http://localhost:8000/docs

2. **Connect Cloud Provider**
   - Go to Settings
   - Add AWS/GCP/Azure credentials
   - Start cost data sync

3. **Explore Features**
   - View cost dashboards
   - Set up budgets
   - Run migration assessment
   - Check optimization recommendations

4. **Configure Alerts**
   - Set budget thresholds
   - Configure notification channels
   - Test alert delivery

---

## 💡 Pro Tips

### Development Mode
```powershell
# Start with hot reload (code changes auto-reload)
docker-compose up
```

### Production Mode
```powershell
# Start with production settings
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Enable Monitoring
```powershell
# Start with monitoring stack
docker-compose --profile monitoring up -d
```

### Enable Logging
```powershell
# Start with ELK stack
docker-compose --profile logging up -d
```

### View Resource Usage
```powershell
docker stats
```

---

## 📞 Need Help?

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Check service status: `docker-compose ps`
3. Review [SETUP_GUIDE.md](SETUP_GUIDE.md)
4. Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
5. Review error messages carefully

---

## ✅ Success!

When you see:
- ✅ All services "Up" and "healthy"
- ✅ Frontend loads at http://localhost:3000
- ✅ API responds at http://localhost:8000

**You're ready to use the Cloud Intelligence Platform!** 🎉

---

## 🎓 What's Next?

- Read [FEATURES_AND_CAPABILITIES.md](FEATURES_AND_CAPABILITIES.md) to learn what you can do
- Check [QUICK_START.md](QUICK_START.md) for usage examples
- Explore the API at http://localhost:8000/docs
- Start optimizing your cloud costs!
