/**
 * Application Configuration
 * Centralized API Base URL for Vercel / Azure Deployments
 */

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const API_ENDPOINTS = {
  HEALTH: `${API_BASE_URL}/health`,
  PRESETS: `${API_BASE_URL}/api/presets`,
  PREDICT: `${API_BASE_URL}/api/predict`,
  PREDICT_BATCH: `${API_BASE_URL}/api/predict-batch`,
  SAMPLE_CSV: `${API_BASE_URL}/api/sample-csv`,
  MODEL_INFO: `${API_BASE_URL}/api/model-info`,
  PLOTS: {
    EDA: `${API_BASE_URL}/plots/eda`,
    EVAL: `${API_BASE_URL}/plots/eval`
  }
};
