/**
 * Brainmate Express Backend
 * Route: /api/health
 */

const express = require('express');
const axios = require('axios');
const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

router.get('/', async (req, res) => {
  let mlHealth = { status: 'unreachable', error: null };

  try {
    const response = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 3000 });
    mlHealth = response.data;
  } catch (err) {
    mlHealth = {
      status: 'offline',
      error: err.code === 'ECONNREFUSED' 
        ? 'ML service not running on ' + ML_SERVICE_URL 
        : err.message,
    };
  }

  const isHealthy = mlHealth.status === 'healthy';

  return res.status(isHealthy ? 200 : 207).json({
    backend: {
      status: 'healthy',
      service: 'Brainmate Express Backend',
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
    },
    ml_service: mlHealth,
  });
});

module.exports = router;

