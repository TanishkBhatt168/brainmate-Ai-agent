'use client';

import React, { useState } from 'react';
import { ExternalLink, Copy, Check, ChevronDown, ChevronUp, Sparkles, Tag, Calendar } from 'lucide-react';

export default function PaperCard({ paper, onClassifyPaper }) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!paper) return null;

  const {
    id = '',
    title = 'Untitled Paper',
    summary = '',
    category = '',
    category_code = '',
    published_date = '',
    similarity_score,
  } = paper;

  const handleCopyId = () => {
    if (!id) return;
    navigator.clipboard.writeText(id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const isLongSummary = summary.length > 220;
  const displaySummary = isExpanded || !isLongSummary ? summary : summary.slice(0, 220) + '...';

  return (
    <div className="rounded-xl border border-surface-border bg-white p-4 shadow-sm hover:shadow transition-all duration-150 flex flex-col gap-2.5 text-left">
      {/* Header: Title & Badges */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-2">
        <h4 className="font-semibold text-charcoal-900 text-sm sm:text-base leading-snug">
          {title}
        </h4>

        {category_code && (
          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-brand-50 text-brand-700 border border-brand-200/60 flex-shrink-0 self-start">
            <Tag className="w-3 h-3" />
            <span>{category_code}</span>
          </span>
        )}
      </div>

      {/* Category Name & Metadata */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-charcoal-500">
        {category && (
          <span className="font-medium text-charcoal-700">
            {category}
          </span>
        )}

        {published_date && (
          <span className="flex items-center gap-1 text-charcoal-400">
            <Calendar className="w-3 h-3" />
            <span>{published_date}</span>
          </span>
        )}

        {id && (
          <button
            onClick={handleCopyId}
            className="flex items-center gap-1 text-charcoal-400 hover:text-charcoal-700 transition-colors"
            title="Copy arXiv ID"
          >
            <span>ID: <code className="font-mono">{id}</code></span>
            {copied ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3" />}
          </button>
        )}

        {typeof similarity_score === 'number' && (
          <span className="text-[11px] text-charcoal-400">
            Match: {(similarity_score * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Summary / Abstract */}
      {summary && (
        <div className="text-xs sm:text-sm text-charcoal-600 leading-relaxed mt-1">
          <p>{displaySummary}</p>
          {isLongSummary && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="mt-1 text-xs font-medium text-brand-600 hover:text-brand-700 flex items-center gap-0.5"
            >
              {isExpanded ? (
                <>
                  <span>Show less</span>
                  <ChevronUp className="w-3 h-3" />
                </>
              ) : (
                <>
                  <span>Read full abstract</span>
                  <ChevronDown className="w-3 h-3" />
                </>
              )}
            </button>
          )}
        </div>
      )}

      {/* Quick Action Buttons */}
      {onClassifyPaper && (
        <div className="pt-2 mt-1 border-t border-surface-border/50 flex items-center justify-end">
          <button
            onClick={() => onClassifyPaper(title, summary)}
            className="inline-flex items-center gap-1.5 text-xs font-medium text-brand-600 hover:text-brand-800 hover:bg-brand-50 px-2.5 py-1 rounded-lg transition-colors"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Predict Category</span>
          </button>
        </div>
      )}
    </div>
  );
}

