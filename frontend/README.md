# Cloud Intelligence Platform - React Frontend

Modern React-based frontend for the Cloud Intelligence Platform.

## Quick Start

### Prerequisites
- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000)

### Troubleshooting

If you encounter module resolution issues:

1. **Clear npm cache:**
   ```bash
   npm cache clean --force
   ```

2. **Delete node_modules and reinstall:**
   ```bash
   rm -rf node_modules package-lock.json
   npm install
   ```

3. **Check Node.js version:**
   ```bash
   node --version  # Should be 18+
   ```

### Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run lint` - Run ESLint

### Features

- ✅ Modern React 18 with TypeScript
- ✅ Material-UI components
- ✅ Responsive design
- ✅ Dark theme
- ✅ Interactive charts
- ✅ Smooth animations
- ✅ Professional UI/UX

### Project Structure

```
src/
├── components/
│   └── Layout/
│       ├── Header.tsx
│       └── Sidebar.tsx
├── pages/
│   ├── Dashboard.tsx
│   ├── Workloads.tsx
│   ├── Costs.tsx
│   ├── Performance.tsx
│   ├── Analytics.tsx
│   └── Settings.tsx
├── App.tsx
└── index.tsx
```

### API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`

Make sure the backend is running before starting the frontend.

### Docker

To run with Docker:

```bash
# Build image
docker build -t cloud-intelligence-frontend .

# Run container
docker run -p 3000:80 cloud-intelligence-frontend
```