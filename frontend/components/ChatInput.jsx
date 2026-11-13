'use client';

import React, { useRef, useEffect } from 'react';
import { ArrowUp, Sparkles } from 'lucide-react';

export default function ChatInput({
  input = '',
  setInput,
  onSend,
  isLoading = false,
  placeholder = 'Ask anything about research papers...',
}) {
  const textareaRef = useRef(null);

  // Auto-resize textarea to fit multiline content up to max height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading && input.trim()) {
        onSend();
      }
    }
  };

  const canSubmit = input.trim().length > 0 && !isLoading;

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4 pt-2">
      <div className="relative flex flex-col bg-white border border-surface-border rounded-2xl shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-500/10 transition-all duration-200">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          rows={1}
          className="w-full resize-none bg-transparent px-4 pt-3.5 pb-12 text-sm sm:text-base text-charcoal-900 placeholder:text-charcoal-400 focus:outline-none min-h-[52px] max-h-[180px] leading-relaxed"
        />

        {/* Bottom toolbar inside input box */}
        <div className="absolute bottom-2.5 right-2.5 flex items-center gap-2">
          <button
            type="button"
            onClick={onSend}
            disabled={!canSubmit}
            className={`flex items-center justify-center w-8 h-8 rounded-full transition-all duration-150 ${
              canSubmit
                ? 'bg-brand-600 text-white hover:bg-brand-700 shadow-sm active:scale-95'
                : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            }`}
            title="Send message (Enter)"
            aria-label="Send message"
          >
            <ArrowUp className="w-4 h-4 stroke-[2.5]" />
          </button>
        </div>
      </div>

      <div className="text-center mt-2 text-[11px] text-charcoal-400">
        Brainmate classifies research with a Logistic Regression model and searches the arXiv dataset.
      </div>
    </div>
  );
}

