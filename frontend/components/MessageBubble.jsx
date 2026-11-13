'use client';

import React from 'react';
import { BookOpen, User, AlertCircle, Copy, Check } from 'lucide-react';
import PaperCard from './PaperCard';
import ClassificationCard from './ClassificationCard';

/**
 * Basic markdown parser to render **bold**, *italic*, `code`, and line breaks
 */
function FormattedText({ content }) {
  if (!content) return null;

  const lines = content.split('\n');

  return (
    <div className="space-y-2 text-sm sm:text-base leading-relaxed">
      {lines.map((line, idx) => {
        if (!line.trim()) {
          return <div key={idx} className="h-2" />;
        }

        // Heading 3
        if (line.startsWith('### ')) {
          return (
            <h3 key={idx} className="text-base sm:text-lg font-semibold text-charcoal-900 pt-1">
              {line.replace('### ', '')}
            </h3>
          );
        }

        // Bullet point
        if (line.startsWith('• ') || line.startsWith('- ')) {
          const text = line.replace(/^[•-]\s+/, '');
          return (
            <div key={idx} className="flex items-start gap-2 ml-1">
              <span className="text-brand-600 font-bold">•</span>
              <span className="flex-1">{renderInlineStyles(text)}</span>
            </div>
          );
        }

        return <p key={idx}>{renderInlineStyles(line)}</p>;
      })}
    </div>
  );
}

function renderInlineStyles(text) {
  // Regex to match **bold**, `code`, and *italic*
  const parts = [];
  let remaining = text;
  let keyIndex = 0;

  while (remaining.length > 0) {
    const boldMatch = remaining.match(/^(.*?)\*\*(.+?)\*\*(.*)/s);
    const codeMatch = remaining.match(/^(.*?)`([^`]+)`(.*)/s);
    const italicMatch = remaining.match(/^(.*?)\*([^*]+)\*(.*)/s);

    let earliest = null;
    let matchType = null;

    if (boldMatch) {
      earliest = boldMatch;
      matchType = 'bold';
    }

    if (codeMatch && (!earliest || codeMatch[1].length < earliest[1].length)) {
      earliest = codeMatch;
      matchType = 'code';
    }

    if (italicMatch && (!earliest || italicMatch[1].length < earliest[1].length)) {
      earliest = italicMatch;
      matchType = 'italic';
    }

    if (!earliest) {
      parts.push(<span key={keyIndex++}>{remaining}</span>);
      break;
    }

    // Add prefix
    if (earliest[1]) {
      parts.push(<span key={keyIndex++}>{earliest[1]}</span>);
    }

    // Add styled text
    if (matchType === 'bold') {
      parts.push(<strong key={keyIndex++} className="font-semibold text-charcoal-900">{earliest[2]}</strong>);
    } else if (matchType === 'code') {
      parts.push(
        <code key={keyIndex++} className="bg-black/5 text-charcoal-800 font-mono text-xs px-1.5 py-0.5 rounded border border-black/5">
          {earliest[2]}
        </code>
      );
    } else if (matchType === 'italic') {
      parts.push(<em key={keyIndex++} className="italic text-charcoal-600">{earliest[2]}</em>);
    }

    remaining = earliest[3];
  }

  return parts;
}

export default function MessageBubble({ message, onClassifyPaper }) {
  const isUser = message.sender === 'user';
  const isError = message.isError;
  const data = message.data || {};
  const papers = data.papers || [];
  const prediction = data.prediction;
  const relatedPapers = data.relatedPapers || [];

  return (
    <div className={`py-4 px-4 sm:px-6 w-full ${isUser ? 'bg-transparent' : 'bg-white/60 border-y border-surface-border/40'}`}>
      <div className="max-w-3xl mx-auto flex items-start gap-3.5">
        {/* Avatar */}
        <div
          className={`w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center flex-shrink-0 shadow-xs text-xs font-semibold ${
            isUser
              ? 'bg-charcoal-800 text-white'
              : isError
              ? 'bg-red-500 text-white'
              : 'bg-brand-600 text-white'
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : isError ? <AlertCircle className="w-4 h-4" /> : <BookOpen className="w-4 h-4" />}
        </div>

        {/* Message Content */}
        <div className="flex-1 min-w-0 space-y-3">
          {/* Sender Header */}
          <div className="flex items-center gap-2">
            <span className="font-medium text-xs text-charcoal-900">
              {isUser ? 'You' : 'Brainmate'}
            </span>
            {message.timestamp && (
              <span className="text-[11px] text-charcoal-400">
                {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            )}
          </div>

          {/* Formatted Text */}
          <div className="text-charcoal-800">
            <FormattedText content={message.text} />
          </div>

          {/* Classification Result Card */}
          {prediction && (
            <ClassificationCard prediction={prediction} />
          )}

          {/* Research Paper Result Cards */}
          {papers.length > 0 && (
            <div className="space-y-3 pt-1">
              {papers.map((paper, idx) => (
                <PaperCard
                  key={paper.id || idx}
                  paper={paper}
                  onClassifyPaper={onClassifyPaper}
                />
              ))}

              {data.coverage_note && (
                <div className="text-[11px] text-charcoal-400 italic pt-1 text-center">
                  ℹ️ {data.coverage_note}
                </div>
              )}
            </div>
          )}

          {/* Related Papers (if returned with classification) */}
          {relatedPapers.length > 0 && (
            <div className="pt-2">
              <div className="text-xs font-medium text-charcoal-500 mb-2">
                Related research papers in dataset:
              </div>
              <div className="space-y-2">
                {relatedPapers.map((paper, idx) => (
                  <PaperCard
                    key={paper.id || idx}
                    paper={paper}
                    onClassifyPaper={onClassifyPaper}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

