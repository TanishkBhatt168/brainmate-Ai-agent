/**
 * Brainmate Express Backend
 * Route: /api/chat
 * 
 * Handles conversational queries by:
 * 1. Discerning intent (Search, Classification, Help / Category Info)
 * 2. Routing to the ML Flask service (TF-IDF search or Logistic Regression classifier)
 * 3. Formulating structured, helpful, non-hallucinatory research responses
 */

const express = require('express');
const axios = require('axios');
const router = express.Router();

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Canonical supported categories for reference
const SUPPORTED_CATEGORIES = {
  'cs.AI': { name: 'Artificial Intelligence', desc: 'Reasoning, knowledge representation, heuristic search, agent architectures.' },
  'cs.CL': { name: 'Computation and Language (NLP)', desc: 'Natural language processing, computational linguistics, language models.' },
  'cs.CR': { name: 'Cryptography and Security', desc: 'Cryptographic algorithms, network security, privacy, adversarial defense.' },
  'cs.CV': { name: 'Computer Vision and Pattern Recognition', desc: 'Image classification, object detection, segmentation, video analysis.' },
  'cs.LG': { name: 'Machine Learning', desc: 'Deep learning, supervised/unsupervised learning, optimization, reinforcement learning.' },
  'cs.RO': { name: 'Robotics', desc: 'Robot kinematics, autonomous navigation, control systems, manipulation.' },
};

/**
 * Extracts Title and Summary from user text if structured tags are present
 */
function extractTitleAndSummary(text) {
  const titleMatch = text.match(/(?:title\s*:\s*)([^\n\r]+)/i);
  const summaryMatch = text.match(/(?:summary|abstract)\s*:\s*([\s\S]+)/i);

  if (titleMatch || summaryMatch) {
    return {
      title: titleMatch ? titleMatch[1].trim() : '',
      summary: summaryMatch ? summaryMatch[1].trim() : text.replace(/(?:title\s*:\s*)[^\n\r]+/i, '').trim(),
    };
  }

  return { title: text, summary: '' };
}

/**
 * Determines whether user query is asking for category prediction
 */
function isClassificationIntent(text) {
  const lower = text.toLowerCase().trim();
  const classifyPatterns = [
    /^(?:classify|predict|categorize)/i,
    /predict(?:\s+the)?\s+category/i,
    /what(?:\s+is)?\s+the\s+category/i,
    /what\s+category/i,
    /which\s+category/i,
    /which\s+field/i,
    /what\s+field/i,
    /classify\s+this\s+paper/i,
    /what\s+is\s+this(?:\s+research)?\s+paper\s+about/i,
    /title\s*:/i,
    /abstract\s*:/i,
  ];

  return classifyPatterns.some((pattern) => pattern.test(lower));
}

/**
 * Checks for greetings, help, or category explanation requests
 */
function isHelpOrCategoryIntent(text) {
  const lower = text.toLowerCase().trim();
  const greetings = ['hi', 'hello', 'hey', 'greetings', 'start', 'help', 'what can you do', 'who are you', 'how do you work'];
  if (greetings.includes(lower) || lower.startsWith('hello') || lower.startsWith('hi brainmate')) {
    return 'greeting';
  }

  if (
    lower.includes('supported categories') ||
    lower.includes('list categories') ||
    lower.includes('what categories') ||
    lower.includes('available categories')
  ) {
    return 'categories';
  }

  return null;
}

/**
 * Strips conversational filler to extract clean research keywords for dataset search
 */
function extractCleanSearchQuery(text) {
  let cleaned = text
    .replace(/^(?:can you\s+)?(?:please\s+)?(?:find|search|show|get|look for|give me)\s+(?:me\s+)?(?:some\s+)?(?:research\s+)?(?:papers|articles|publications|works)?\s*(?:about|on|related to|regarding|for)?/i, '')
    .replace(/^(?:tell me about|what are the papers on|search for)\s+/i, '')
    .trim();

  // Strip trailing punctuation
  cleaned = cleaned.replace(/[?.!]+$/, '').trim();

  return cleaned || text.trim();
}

router.post('/', async (req, res) => {
  const { message = '' } = req.body || {};
  const userText = typeof message === 'string' ? message.trim() : '';

  if (!userText) {
    return res.status(400).json({
      error: 'Invalid Request',
      message: 'Message cannot be empty.',
    });
  }

  // 1. Check for greeting or category guide intent
  const helpType = isHelpOrCategoryIntent(userText);
  if (helpType === 'greeting') {
    return res.status(200).json({
      success: true,
      type: 'general',
      message:
        "Hello! I'm **Brainmate**, your scientific research assistant powered by an arXiv dataset and a trained Logistic Regression model.\n\n" +
        "Here is what I can do for you:\n\n" +
        "• **Search Papers:** Find scientific papers on AI, Machine Learning, Computer Vision, Robotics, NLP, and Security.\n" +
        "• **Classify Papers:** Provide a paper title or abstract to predict its official arXiv category and confidence score.\n" +
        "• **Explore Categories:** Learn about the 6 core arXiv categories indexed in the system.\n\n" +
        "*Try asking:* \"*Find research papers about Computer Vision*\" or \"*Classify: Transformers for Language Modeling*\".",
      data: null,
    });
  }

  if (helpType === 'categories') {
    let catList = 'Brainmate is trained and indexed across **6 core arXiv Computer Science & AI domains**:\n\n';
    Object.entries(SUPPORTED_CATEGORIES).forEach(([code, info]) => {
      catList += `• **\`${code}\` — ${info.name}**\n  ${info.desc}\n\n`;
    });
    catList += 'You can ask me to search for papers within any of these domains or paste a title/abstract to classify it.';

    return res.status(200).json({
      success: true,
      type: 'general',
      message: catList,
      data: { categories: SUPPORTED_CATEGORIES },
    });
  }

  // 2. Check for classification / category prediction intent
  if (isClassificationIntent(userText)) {
    // Extract title & summary or clean prefix
    let cleanForPrediction = userText
      .replace(/^(?:please\s+)?(?:predict|classify|categorize)(?:\s+the)?(?:\s+category\s+of)?(?:\s+this\s+paper)?(?:\s*:)?/i, '')
      .replace(/^what\s+(?:category|is\s+the\s+category|is\s+this\s+paper\s+about)\s*(?:of|for)?(?:\s*:)?/i, '')
      .trim();

    const { title, summary } = extractTitleAndSummary(cleanForPrediction || userText);

    try {
      const predRes = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        { title, summary },
        { timeout: 8000 }
      );

      const prediction = predRes.data;

      // Also retrieve 2 related papers from the dataset to enhance the answer
      let relatedPapers = [];
      try {
        const searchSample = (title || summary).slice(0, 120);
        const searchRes = await axios.post(
          `${ML_SERVICE_URL}/search`,
          { query: searchSample, top_k: 3 },
          { timeout: 5000 }
        );
        relatedPapers = (searchRes.data && searchRes.data.papers) || [];
      } catch (e) {
        // Optional enrichment; ignore failure
      }

      const formattedText =
        `### 🔬 Research Category Prediction\n\n` +
        `• **Predicted Category:** **${prediction.predicted_category}** (\`${prediction.category_code}\`)\n` +
        `• **Model Confidence:** **${prediction.confidence_percentage}**\n\n` +
        `*Evaluated with the pre-trained Logistic Regression + TF-IDF model on arXiv scientific literature.*`;

      return res.status(200).json({
        success: true,
        type: 'classification',
        message: formattedText,
        data: {
          prediction,
          relatedPapers,
        },
      });
    } catch (err) {
      console.error('Flask prediction failed:', err.message);
      const msg = err.response?.data?.message || 'Unable to classify paper. Please check ML service connection.';
      return res.status(503).json({
        success: false,
        type: 'error',
        message: `⚠️ **Classification Error:** ${msg}`,
      });
    }
  }

  // 3. Default to Research Paper Search
  const searchQuery = extractCleanSearchQuery(userText);

  try {
    const searchRes = await axios.post(
      `${ML_SERVICE_URL}/search`,
      { query: searchQuery, top_k: 6 },
      { timeout: 10000 }
    );

    const result = searchRes.data;
    const count = result.papers ? result.papers.length : 0;

    let responseMessage = '';
    if (count > 0) {
      responseMessage = `Found **${count} relevant research papers** for **"${searchQuery}"** in the arXiv dataset:`;
    } else {
      responseMessage = `No papers directly matched the query **"${searchQuery}"**. Try searching for broader technical terms such as *Neural Networks*, *Computer Vision*, *Robotics*, *Transformers*, or *Cryptography*.`;
    }

    return res.status(200).json({
      success: true,
      type: 'search',
      message: responseMessage,
      data: {
        query: searchQuery,
        papers: result.papers || [],
        total_results: count,
        coverage_note: result.coverage_note,
      },
    });
  } catch (err) {
    console.error('Flask search failed:', err.message);
    const msg = err.response?.data?.message || 'The ML search service is currently unavailable.';
    return res.status(503).json({
      success: false,
      type: 'error',
      message: `⚠️ **Search Error:** ${msg}`,
    });
  }
});

module.exports = router;

