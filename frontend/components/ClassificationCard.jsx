'use client';

import React, { useState } from 'react';
import { Sparkles, ChevronDown, ChevronUp, CheckCircle2, BarChart2 } from 'lucide-react';

export default function ClassificationCard({ prediction }) {
  const [showAllProbabilities, setShowAllProbabilities] = useState(false);

  if (!prediction) return null;

  const {
    predicted_category = '',
    category_code = '',
    confidence = 0,
    confidence_percentage = 'N/A',
    ranked_predictions = [],
    input_preview = {},
  } = prediction;

  const confidenceValue = typeof confidence === 'number' ? confidence * 100 : 0;

  return (
    <div className="rounded-xl border border-brand-200/80 bg-brand-50/30 p-4 shadow-sm text-left flex flex-col gap-3 my-2">
      {/* Primary Result Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-brand-100">
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-brand-600 text-white shadow-sm flex-shrink-0">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-semibold text-brand-700 uppercase tracking-wider">
              Predicted Category
            </div>
            <div className="text-base font-semibold text-charcoal-900">
              {predicted_category} <span className="text-brand-600">({category_code})</span>
            </div>
          </div>
        </div>

        {/* Confidence Badge */}
        <div className="flex items-center gap-2 self-start sm:self-auto bg-white px-3 py-1 rounded-full border border-brand-200 shadow-xs">
          <span className="text-xs text-charcoal-500">Confidence:</span>
          <span className="text-xs font-semibold text-brand-700">{confidence_percentage}</span>
        </div>
      </div>

      {/* Confidence Visual Bar */}
      {confidenceValue > 0 && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-charcoal-500">
            <span>Model certainty</span>
            <span className="font-medium text-charcoal-700">{confidence_percentage}</span>
          </div>
          <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
            <div
              className="bg-brand-600 h-full rounded-full transition-all duration-500 ease-out"
              style={{ width: `${Math.min(confidenceValue, 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Ranked Probability Breakdown */}
      {ranked_predictions.length > 0 && (
        <div className="mt-1">
          <button
            onClick={() => setShowAllProbabilities(!showAllProbabilities)}
            className="flex items-center gap-1.5 text-xs font-medium text-brand-700 hover:text-brand-900 transition-colors"
          >
            <BarChart2 className="w-3.5 h-3.5" />
            <span>{showAllProbabilities ? 'Hide probability breakdown' : 'View all 6 category probabilities'}</span>
            {showAllProbabilities ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
          </button>

          {showAllProbabilities && (
            <div className="mt-2.5 space-y-1.5 bg-white p-3 rounded-lg border border-brand-100 text-xs">
              {ranked_predictions.map((item, idx) => {
                const probVal = (item.probability || 0) * 100;
                const isTop = idx === 0;
                return (
                  <div key={item.category_code} className="flex items-center gap-2">
                    <span className="font-mono text-charcoal-500 w-12 flex-shrink-0">
                      {item.category_code}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="flex justify-between text-[11px] mb-0.5">
                        <span className={`truncate ${isTop ? 'font-semibold text-charcoal-900' : 'text-charcoal-600'}`}>
                          {item.category_name}
                        </span>
                        <span className={`font-mono ${isTop ? 'font-semibold text-brand-700' : 'text-charcoal-500'}`}>
                          {item.percentage}
                        </span>
                      </div>
                      <div className="w-full bg-gray-100 h-1.5 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${isTop ? 'bg-brand-600' : 'bg-gray-400'}`}
                          style={{ width: `${Math.min(probVal, 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Input preview if available */}
      {input_preview.title && (
        <div className="text-[11px] text-charcoal-400 border-t border-brand-100/60 pt-2 truncate">
          Analyzed paper: <span className="italic text-charcoal-600">"{input_preview.title}"</span>
        </div>
      )}
    </div>
  );
}

