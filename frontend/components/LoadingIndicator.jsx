'use client';

import React from 'react';
import { BookOpen } from 'lucide-react';

export default function LoadingIndicator() {
  return (
    <div className="flex items-start gap-3.5 my-4 max-w-3xl mx-auto w-full px-4 animate-fade-in">
      <div className="w-8 h-8 rounded-lg bg-brand-600 text-white flex items-center justify-center flex-shrink-0 shadow-sm mt-0.5">
        <BookOpen className="w-4 h-4 animate-pulse" />
      </div>

      <div className="flex items-center gap-2 bg-white border border-surface-border rounded-2xl px-4 py-3 shadow-xs">
        <span className="text-xs sm:text-sm font-medium text-charcoal-600">
          Brainmate is researching
        </span>
        <div className="flex items-center gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-brand-600 animate-bounce [animation-delay:-0.3s]" />
          <span className="w-1.5 h-1.5 rounded-full bg-brand-600 animate-bounce [animation-delay:-0.15s]" />
          <span className="w-1.5 h-1.5 rounded-full bg-brand-600 animate-bounce" />
        </div>
      </div>
    </div>
  );
}

