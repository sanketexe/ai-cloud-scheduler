# Demo Images Directory

This directory contains screenshots and demo images for the Cloud Intelligence FinOps Platform.

## 📸 Required Screenshots

Please add the following screenshots to showcase the platform:

### 1. **dashboard-overview.png**
- Main dashboard showing cost overview
- Multi-cloud spending visualization
- Budget alerts and notifications
- **Recommended size**: 1920x1080 or 1440x900

### 2. **cost-analysis.png**
- Cost analysis page with charts
- Historical trends (90-day view)
- Service breakdown
- Optimization recommendations
- **Recommended size**: 1920x1080 or 1440x900

### 3. **migration-wizard.png**
- Migration wizard 4-step process
- Organization profile form
- Progress indicator
- **Recommended size**: 1920x1080 or 1440x900

### 4. **migration-dashboard.png**
- Migration projects overview
- Timeline and progress tracking
- Phase-by-phase status
- **Recommended size**: 1920x1080 or 1440x900

### 5. **resource-organization.png**
- Resource discovery interface
- Tagging and categorization
- Team-based organization
- **Recommended size**: 1920x1080 or 1440x900

### 6. **provider-recommendations.png**
- AWS vs GCP vs Azure comparison
- Scoring matrix (cost, compatibility, compliance, performance)
- Migration recommendations
- **Recommended size**: 1920x1080 or 1440x900

### 7. **grafana-monitoring.png**
- Grafana dashboard
- System metrics and monitoring
- Performance graphs
- **Recommended size**: 1920x1080 or 1440x900

### 8. **api-documentation.png**
- Swagger/OpenAPI documentation
- Interactive API testing
- Endpoint listings
- **Recommended size**: 1920x1080 or 1440x900

## 📝 Screenshot Guidelines

### **Quality Standards**
- ✅ High resolution (minimum 1440x900)
- ✅ Clear, readable text
- ✅ Professional appearance
- ✅ Consistent browser/theme
- ✅ No personal/sensitive data

### **Content Guidelines**
- ✅ Use realistic demo data
- ✅ Show key features prominently
- ✅ Include relevant UI elements
- ✅ Demonstrate value proposition
- ✅ Professional color scheme

### **File Naming**
- Use exact names listed above
- PNG format preferred
- Keep file sizes reasonable (<2MB each)

## 🚀 How to Take Screenshots

### **Recommended URLs**
```bash
# Start the platform first
docker-compose up -d

# Then capture these pages:
http://localhost:3000                    # Main dashboard
http://localhost:3000/cost-analysis     # Cost analysis
http://localhost:3000/migration-wizard  # Migration wizard
http://localhost:3000/migration         # Migration dashboard
http://localhost:3000/resources         # Resource organization
http://localhost:3000/recommendations   # Provider recommendations
http://localhost:3001                   # Grafana (admin/admin)
http://localhost:8000/docs              # API documentation
```

### **Browser Settings**
- Use Chrome or Firefox
- Set zoom to 100%
- Hide bookmarks bar
- Use incognito/private mode for clean appearance
- Recommended window size: 1440x900 or 1920x1080

## 📁 File Structure
```
docs/images/demo/
├── dashboard-overview.png
├── cost-analysis.png
├── migration-wizard.png
├── migration-dashboard.png
├── resource-organization.png
├── provider-recommendations.png
├── grafana-monitoring.png
├── api-documentation.png
└── README.md (this file)
```

## 🎯 After Adding Images

Once you add the images:
1. Commit and push to GitHub
2. The demo page will automatically display them
3. Share the demo link: `https://github.com/sanketexe/ai-cloud-scheduler/blob/main/docs/DEMO.md`

---

**Ready to showcase your amazing platform! 🚀**