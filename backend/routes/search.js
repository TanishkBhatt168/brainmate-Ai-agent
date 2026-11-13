/**
 * Brainmate Express Backend
 * Route: /api/search
 */

const express = require('express');
const axios = require('axios');
const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

router.post('/', async (req, res) => {
  const { query = '', top_k = 5 } = req.body || {};

  const cleanQuery = typeof query === 'string' ? query.trim() : '';

  if (!cleanQuery) {
    return res.status(400).json({
      error: 'Invalid Request',
      message: 'Please provide a non-empty search query.',
    });
  }

  try {
    const response = await axios.post(
      `${ML_SERVICE_URL}/search`,
      { query: cleanQuery, top_k: parseInt(top_k, 10) || 5 },
      { timeout: 10000 }
    );

    return res.status(200).json({
      success: true,
      data: response.data,
    });
  } catch (err) {
    console.error('Search error:', err.message);
    const status = err.response ? err.response.status : 503;
    const msg = err.response && err.response.data && err.response.data.message
      ? err.response.data.message
      : 'ML Service search is currently unavailable. Please verify the Flask service is running.';

    return res.status(status).json({
      error: 'Search Failed',
      message: msg,
    });
  }
});

module.exports = router;

