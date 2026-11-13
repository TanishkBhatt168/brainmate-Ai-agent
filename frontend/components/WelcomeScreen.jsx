'use client';

import React from 'react';
import { BookOpen, Search, Tag, Cpu, ShieldCheck } from 'lucide-react';

const SUGGESTIONS = [
  {
    icon: Search,
    title: 'Computer Vision',
    subtitle: 'Find papers on vision transformers & segmentation',
    prompt: 'Find research papers about Computer Vision and image segmentation',
  },
  {
    icon: Cpu,
    title: 'Machine Learning',
    subtitle: 'Explore deep learning & optimization papers',
    prompt: 'Show papers related to Machine Learning and deep neural networks',
  },
  {
    icon: Tag,
    title: 'Classify a Paper',
    subtitle: 'Predict arXiv category with trained model',
    prompt: 'Classify this paper:\nTitle: Attention Is All You Need\nAbstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.',
  },
  {
    icon: ShieldCheck,
    title: 'Cryptography & Security',
    subtitle: 'Discover papers on security & encryption',
    prompt: 'Find research papers about Cryptography and network security',
  },
];

export default function WelcomeScreen({ onSelectPrompt }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center p-6 text-center max-w-2xl mx-auto my-auto animate-fade-in">
      {/* Icon & App Name */}
      <div className="flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-tr from-brand-600 to-brand-500 text-white shadow-md shadow-brand-500/20 mb-5">
        <BookOpen className="w-7 h-7" />
      </div>

      {/* Main Heading */}
      <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-charcoal-900 mb-2">
        What would you like to research today?
      </h1>

      {/* Subtitle */}
      <p className="text-sm sm:text-base text-charcoal-500 mb-8 max-w-md">
        Explore research papers and discover insights with Brainmate.
      </p>

      {/* Quick Prompt Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
        {SUGGESTIONS.map((item, idx) => {
          const Icon = item.icon;
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.prompt)}
              className="p-3.5 rounded-xl border border-surface-border bg-white hover:bg-gray-50/80 hover:border-brand-300 hover:shadow-sm text-left transition-all duration-150 flex items-start gap-3 group"
            >
              <div className="p-2 rounded-lg bg-surface-subtle text-charcoal-600 group-hover:text-brand-600 group-hover:bg-brand-50 transition-colors flex-shrink-0">
                <Icon className="w-4 h-4" />
              </div>
              <div className="min-w-0">
                <div className="text-sm font-medium text-charcoal-800 group-hover:text-brand-700 transition-colors truncate">
                  {item.title}
                </div>
                <div className="text-xs text-charcoal-500 truncate mt-0.5">
                  {item.subtitle}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

