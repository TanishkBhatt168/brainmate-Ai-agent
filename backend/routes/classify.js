/**
 * Brainmate Express Backend
 * Route: /api/classify
 */

const express = require('express');
const axios = require('axios');
const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

router.post('/', async (req, res) => {
  const { title = '', summary = '' } = req.body || {};

  const cleanTitle = typeof title === 'string' ? title.trim() : '';
  const cleanSummary = typeof summary === 'string' ? summary.trim() : '';

  if (!cleanTitle && !cleanSummary) {
    return res.status(400).json({
      error: 'Invalid Request',
      message: 'Please provide either a research paper title or summary (or both) to classify.',
    });
  }

  try {
    const response = await axios.post(
      `${ML_SERVICE_URL}/predict`,
      { title: cleanTitle, summary: cleanSummary },
      { timeout: 8000 }
    );

    return res.status(200).json({
      success: true,
      data: response.data,
    });
  } catch (err) {
    console.error('Classification error:', err.message);
    const status = err.response ? err.response.status : 503;
    const msg = err.response && err.response.data && err.response.data.message
      ? err.response.data.message
      : 'ML Service is currently unavailable. Please verify that the Flask service is running on ' + ML_SERVICE_URL;

    return res.status(status).json({
      error: 'Classification Failed',
      message: msg,
    });
  }
});

module.exports = router;

