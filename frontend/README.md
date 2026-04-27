# Frontend — ECG Anomaly Detection

Mobile-first web app. React 18 + Vite + Tailwind CSS + React Router + Plotly.

## Install & run

```bash
npm install       # First time only
npm run dev       # http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://localhost:8000` (the FastAPI
backend) — no CORS config needed during development.

## Build for production

```bash
npm run build     # outputs ./dist
npm run preview   # serves ./dist on :4173 for smoke-testing
```

## Configuration

Override the API base URL via a Vite env var:

```bash
VITE_API_BASE_URL=https://ecg-api.example.com npm run build
```

## Screens

- `/` — Upload page with dropzone and "Simulate demo recording" button
- `/report/:id` — Analysis report: overall status, disclaimer, per-lead SQI,
  12-lead ECG viewer (when samples are in state), sorted condition cards,
  download JSON, share link
- `/history` — List of past reports with top finding and tier badge

## Dark mode

Click the moon/sun icon in the header. Preference persists in `localStorage`
under `ecg.theme`.

## File map

```
src/
  main.jsx                  React entry point
  App.jsx                   Router
  index.css                 Tailwind + design tokens
  components/
    Layout.jsx              App shell + nav
    EcgPlot.jsx             Plotly-based single / 12-lead viewer
    ConditionCard.jsx       Condition result card with tier badge
    LeadQualityBadge.jsx    Per-lead SQI indicator
    Skeleton.jsx            Loading placeholders
    ErrorBanner.jsx         Error + per-lead SQI breakdown
  pages/
    UploadPage.jsx          Screen 1
    ReportPage.jsx          Screen 3 (result)
    HistoryPage.jsx         Screen 4
  services/
    api.js                  Centralized axios client
  utils/
    riskTiers.js            Tier colors, condition descriptions, formatters
    synthEcg.js             Client-side synthetic ECG for demo mode
  hooks/
    useDarkMode.js          Dark mode toggle backed by localStorage
```
