/**
 * =============================================================================
 * Brainmate - Main Backend Server
 * File: backend/server.js
 * Technology: Node.js + Express
 * Default Port: 5000
 * =============================================================================
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');

// Route imports
const healthRoutes = require('./routes/health');
const chatRoutes = require('./routes/chat');
const searchRoutes = require('./routes/search');
const classifyRoutes = require('./routes/classify');
const paperRoutes = require('./routes/paper');

const app = express();
const PORT = process.env.PORT || 5000;
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Middleware
app.use(cors({
  origin: '*', // Allow all origins for dev flexibility (Next.js at port 3000)
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));

app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: true, limit: '5mb' }));

// Request logging in development
app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${duration}ms)`);
  });
  next();
});

// API Routes
app.use('/api/health', healthRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/search', searchRoutes);
app.use('/api/classify', classifyRoutes);
app.use('/api/paper', paperRoutes);

// Root route
app.get('/', (req, res) => {
  res.json({
    service: 'Brainmate Express Backend',
    version: '1.0.0',
    status: 'running',
    endpoints: [
      'GET  /api/health',
      'POST /api/chat',
      'POST /api/search',
      'POST /api/classify',
      'GET  /api/paper/:id',
    ],
    ml_service_target: ML_SERVICE_URL,
  });
});

// 404 Handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Cannot ${req.method} ${req.originalUrl}`,
  });
});

// Global Error Handler
app.use((err, req, res, next) => {
  console.error('[Global Error]', err);
  res.status(err.status || 500).json({
    error: 'Internal Server Error',
    message: err.message || 'An unexpected error occurred on the server.',
  });
});

// Start Server
app.listen(PORT, () => {
  console.log('====================================================');
  console.log(`🚀 Brainmate Express Backend running on http://localhost:${PORT}`);
  console.log(`🔗 Target Flask ML Service: ${ML_SERVICE_URL}`);
  console.log('====================================================');
});

