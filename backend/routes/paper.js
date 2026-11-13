/**
 * Brainmate Express Backend
 * Route: /api/paper/:id
 */

const express = require('express');
const axios = require('axios');
const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

router.get('/:id', async (req, res) => {
  const paperId = req.params.id;

  if (!paperId) {
    return res.status(400).json({
      error: 'Invalid Request',
      message: 'Paper ID is required.',
    });
  }

  try {
    const response = await axios.get(
      `${ML_SERVICE_URL}/paper/${encodeURIComponent(paperId)}`,
      { timeout: 5000 }
    );

    return res.status(200).json({
      success: true,
      paper: response.data.paper,
    });
  } catch (err) {
    const status = err.response ? err.response.status : 503;
    const msg = err.response && err.response.data && err.response.data.message
      ? err.response.data.message
      : 'Paper lookup failed or service is offline.';

    return res.status(status).json({
      error: 'Paper Lookup Failed',
      message: msg,
    });
  }
});

module.exports = router;

