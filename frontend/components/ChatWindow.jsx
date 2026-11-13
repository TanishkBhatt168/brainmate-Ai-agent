'use client';

import React, { useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';
import LoadingIndicator from './LoadingIndicator';
import WelcomeScreen from './WelcomeScreen';

export default function ChatWindow({
  messages = [],
  isLoading = false,
  onSelectPrompt,
  onClassifyPaper,
}) {
  const messagesEndRef = useRef(null);

  // Auto-scroll to latest message on updates
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  if (messages.length === 0) {
    return <WelcomeScreen onSelectPrompt={onSelectPrompt} />;
  }

  return (
    <div className="flex-1 overflow-y-auto flex flex-col">
      <div className="flex-1">
        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onClassifyPaper={onClassifyPaper}
          />
        ))}

        {isLoading && <LoadingIndicator />}

        <div ref={messagesEndRef} className="h-4" />
      </div>
    </div>
  );
}

