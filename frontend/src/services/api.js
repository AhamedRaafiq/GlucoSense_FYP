import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const subjectService = {
  getSubjects: () => api.get('/subjects/').then(r => r.data),
  createSubject: (data) => api.post('/subjects/', data).then(r => r.data),
  getSubject: (id) => api.get(`/subjects/${id}`).then(r => r.data),
  deleteSubject: (id) => api.delete(`/subjects/${id}`).then(r => r.data),
};

export const dataService = {
  uploadCSV: (subjectId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post(`/data/upload?subject_id=${subjectId}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }).then(r => r.data);
  },
  listSerialPorts: () => 
    api.get('/data/serial/ports').then(r => r.data),
  startSerial: (subjectId, port = 'COM7', simulate = false) => 
    api.post(`/data/serial/start?subject_id=${subjectId}&port=${port}&simulate=${simulate}`).then(r => r.data),
  stopSerial: (subjectId) => 
    api.post(`/data/serial/stop?subject_id=${subjectId}`).then(r => r.data),
  getSerialStatus: (subjectId) => 
    api.get(`/data/serial/status/${subjectId}`).then(r => r.data),
  getLiveStreamUrl: (subjectId) => 
    `${window.location.origin}/api/data/serial/stream/${subjectId}`,
  getWebSocketUrl: (subjectId) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname;
    return `${protocol}//${host}:8000/api/data/serial/ws/${subjectId}`;
  },
};

export const pipelineService = {
  runPipeline: (subjectId) => api.post(`/pipeline/run/${subjectId}`).then(r => r.data),
  getPipelineRunStatus: (runId) => api.get(`/pipeline/status/${runId}`).then(r => r.data),
};

export const predictionService = {
  getPredictions: (subjectId = null) => {
    const url = subjectId ? `/predictions/?subject_id=${subjectId}` : '/predictions/';
    return api.get(url).then(r => r.data);
  },
  getPredictionDetail: (id) => api.get(`/predictions/${id}`).then(r => r.data),
};

export const dashboardService = {
  getStats: () => api.get('/dashboard/stats').then(r => r.data),
  getDistribution: () => api.get('/dashboard/distribution').then(r => r.data),
  getPredictedVsActual: () => api.get('/dashboard/predicted-vs-actual').then(r => r.data),
  getFeatureImportance: () => api.get('/dashboard/feature-importance').then(r => r.data),
  retrainModel: () => api.post('/dashboard/model/retrain').then(r => r.data),
};

export default api;
