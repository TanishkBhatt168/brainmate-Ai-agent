'use client';

import React, { useState, useEffect } from 'react';
import { Menu, Plus, Sparkles, BookOpen, AlertCircle } from 'lucide-react';
import Sidebar from '../components/Sidebar';
import ChatWindow from '../components/ChatWindow';
import ChatInput from '../components/ChatInput';

const STORAGE_KEY = 'brainmate_conversations_v1';
const CURRENT_CHAT_KEY = 'brainmate_current_chat_id_v1';
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:5000';

function generateId() {
  return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

export default function Home() {
  const [conversations, setConversations] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [isClient, setIsClient] = useState(false);

  // Initialize from localStorage on mount
  useEffect(() => {
    setIsClient(true);
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      const savedChatId = localStorage.getItem(CURRENT_CHAT_KEY);

      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setConversations(parsed);
          const activeExists = parsed.some((c) => c.id === savedChatId);
          setCurrentChatId(activeExists ? savedChatId : parsed[0].id);
          return;
        }
      }

      // Initialize with fresh chat if none found
      const initialChat = {
        id: generateId(),
        title: 'New Research Chat',
        createdAt: new Date().toISOString(),
        messages: [],
      };
      setConversations([initialChat]);
      setCurrentChatId(initialChat.id);
    } catch (err) {
      console.error('Failed to load conversations from localStorage:', err);
    }
  }, []);

  // Save conversations to localStorage
  useEffect(() => {
    if (!isClient) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
      if (currentChatId) {
        localStorage.setItem(CURRENT_CHAT_KEY, currentChatId);
      }
    } catch (err) {
      console.error('Failed to save to localStorage:', err);
    }
  }, [conversations, currentChatId, isClient]);

  // Periodic health check with backend
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/health`);
        const data = await res.json();
        setSystemStatus(data);
        if (data.ml_service?.status !== 'healthy') {
          setErrorMessage('ML Service offline: Research prediction and search might be limited.');
        } else {
          setErrorMessage(null);
        }
      } catch (e) {
        setSystemStatus({ backend: { status: 'offline' } });
        setErrorMessage('Cannot reach Brainmate backend. Ensure server is running on port 5000.');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const currentChat = conversations.find((c) => c.id === currentChatId) || { messages: [] };

  // New Chat handler
  const handleNewChat = () => {
    const newChat = {
      id: generateId(),
      title: 'New Research Chat',
      createdAt: new Date().toISOString(),
      messages: [],
    };
    setConversations((prev) => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    setMobileSidebarOpen(false);
  };

  // Select Chat handler
  const handleSelectChat = (id) => {
    setCurrentChatId(id);
    setMobileSidebarOpen(false);
  };

  // Delete Chat handler
  const handleDeleteChat = (id) => {
    setConversations((prev) => {
      const remaining = prev.filter((c) => c.id !== id);
      if (remaining.length === 0) {
        const fresh = {
          id: generateId(),
          title: 'New Research Chat',
          createdAt: new Date().toISOString(),
          messages: [],
        };
        setCurrentChatId(fresh.id);
        return [fresh];
      }
      if (currentChatId === id) {
        setCurrentChatId(remaining[0].id);
      }
      return remaining;
    });
  };

  // Send message
  const handleSendMessage = async (customText = null) => {
    const textToSend = (customText !== null ? customText : input).trim();
    if (!textToSend || isLoading) return;

    setInput('');

    // Ensure we have an active chat
    let chatId = currentChatId;
    if (!chatId) {
      const newChat = {
        id: generateId(),
        title: textToSend.slice(0, 32) + (textToSend.length > 32 ? '...' : ''),
        createdAt: new Date().toISOString(),
        messages: [],
      };
      setConversations((prev) => [newChat, ...prev]);
      setCurrentChatId(newChat.id);
      chatId = newChat.id;
    }

    const userMsg = {
      id: 'msg_' + Date.now(),
      sender: 'user',
      text: textToSend,
      timestamp: new Date().toISOString(),
    };

    // Update conversation with user message and update chat title if first message
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id === chatId) {
          const isFirst = c.messages.length === 0;
          return {
            ...c,
            title: isFirst ? textToSend.slice(0, 32) + (textToSend.length > 32 ? '...' : '') : c.title,
            messages: [...c.messages, userMsg],
          };
        }
        return c;
      })
    );

    setIsLoading(true);

    try {
      const res = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: textToSend }),
      });

      const data = await res.json();

      const assistantMsg = {
        id: 'msg_' + (Date.now() + 1),
        sender: 'assistant',
        text: data.message || 'I processed your request.',
        timestamp: new Date().toISOString(),
        type: data.type || 'general',
        data: data.data || {},
        isError: !data.success && res.status !== 200,
      };

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id === chatId) {
            return {
              ...c,
              messages: [...c.messages, assistantMsg],
            };
          }
          return c;
        })
      );
    } catch (err) {
      console.error('Chat error:', err);
      const errorMsg = {
        id: 'msg_' + (Date.now() + 1),
        sender: 'assistant',
        text: '⚠️ **Backend Connection Error:** Brainmate could not connect to the Express server. Please make sure the backend is running on `http://localhost:5000`.',
        timestamp: new Date().toISOString(),
        isError: true,
      };

      setConversations((prev) =>
        prev.map((c) => (c.id === chatId ? { ...c, messages: [...c.messages, errorMsg] } : c))
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Classify a specific paper
  const handleClassifyPaper = (title, summary) => {
    const prompt = `Classify this paper:\nTitle: ${title}\nAbstract: ${summary}`;
    handleSendMessage(prompt);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#fbfbfb]">
      {/* Desktop & Mobile Sidebar */}
      <Sidebar
        conversations={conversations}
        currentChatId={currentChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        isMobile={false}
        systemStatus={systemStatus}
      />

      {/* Mobile Drawer */}
      <div className="lg:hidden">
        <Sidebar
          conversations={conversations}
          currentChatId={currentChatId}
          onSelectChat={handleSelectChat}
          onNewChat={handleNewChat}
          onDeleteChat={handleDeleteChat}
          isOpen={mobileSidebarOpen}
          onToggle={() => setMobileSidebarOpen(false)}
          isMobile={true}
          systemStatus={systemStatus}
        />
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Navbar */}
        <header className="h-14 border-b border-surface-border/60 bg-white/80 backdrop-blur-sm px-4 flex items-center justify-between flex-shrink-0 z-10">
          <div className="flex items-center gap-3">
            {/* Mobile menu button */}
            <button
              onClick={() => setMobileSidebarOpen(true)}
              className="lg:hidden p-1.5 text-charcoal-600 hover:text-charcoal-900 hover:bg-black/5 rounded-md transition-colors"
              aria-label="Open menu"
            >
              <Menu className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-2">
              <span className="font-semibold text-charcoal-900 text-sm sm:text-base tracking-tight truncate max-w-[200px] sm:max-w-xs">
                {currentChat.title || 'Brainmate'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleNewChat}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-charcoal-700 hover:text-charcoal-900 hover:bg-black/5 rounded-lg border border-surface-border shadow-xs transition-colors"
              title="Start a new chat"
            >
              <Plus className="w-3.5 h-3.5 text-brand-600" />
              <span className="hidden sm:inline">New Chat</span>
            </button>
          </div>
        </header>

        {/* Global Connection Warning Banner */}
        {errorMessage && (
          <div className="bg-amber-50 border-b border-amber-200 px-4 py-2 text-xs text-amber-800 flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 flex-shrink-0" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="text-amber-600 hover:text-amber-800 text-[11px] font-semibold"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Chat Messages Body */}
        <ChatWindow
          messages={currentChat.messages || []}
          isLoading={isLoading}
          onSelectPrompt={(p) => handleSendMessage(p)}
          onClassifyPaper={handleClassifyPaper}
        />

        {/* Floating Chat Input */}
        <ChatInput
          input={input}
          setInput={setInput}
          onSend={() => handleSendMessage()}
          isLoading={isLoading}
          placeholder="Ask anything about research papers..."
        />
      </div>
    </div>
  );
}

