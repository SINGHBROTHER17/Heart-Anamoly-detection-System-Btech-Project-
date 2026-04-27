import axios from 'axios';

/**
 * Centralized API client.
 *
 * In dev, Vite proxies /api/* to http://localhost:8000 (vite.config.js).
 * In production, override via VITE_API_BASE_URL env var.
 */
const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

const client = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// Response interceptor: normalize errors so UI has a consistent shape.
client.interceptors.response.use(
  (r) => r,
  (error) => {
    if (error.response) {
      const { status, data } = error.response;
      const normalized = new Error(
        data?.detail || data?.error || `API error ${status}`,
      );
      normalized.status = status;
      normalized.error = data?.error;
      normalized.perLeadSqi = data?.per_lead_sqi;
      return Promise.reject(normalized);
    }
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timed out. Please try again.'));
    }
    return Promise.reject(new Error('Network error. Please check your connection.'));
  },
);

/** POST /analyze — CSV file upload */
export async function analyzeCsv(file, { sampleRate = 500, patientId } = {}) {
  const form = new FormData();
  form.append('file', file);
  form.append('sample_rate', String(sampleRate));
  if (patientId) form.append('patient_id', patientId);

  const { data } = await client.post('/analyze', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

/** POST /analyze/json — per-lead JSON payload */
export async function analyzeJson(leads, { patientId } = {}) {
  const { data } = await client.post('/analyze/json', {
    leads,
    patient_id: patientId,
  });
  return data;
}

/** GET /report/{id} */
export async function getReport(reportId) {
  const { data } = await client.get(`/report/${encodeURIComponent(reportId)}`);
  return data;
}

/** GET /reports */
export async function listReports(limit = 50) {
  const { data } = await client.get('/reports', { params: { limit } });
  return data;
}

/** POST /feedback */
export async function submitFeedback(feedback) {
  const { data } = await client.post('/feedback', feedback);
  return data;
}

/** GET /health */
export async function getHealth() {
  const { data } = await client.get('/health');
  return data;
}
