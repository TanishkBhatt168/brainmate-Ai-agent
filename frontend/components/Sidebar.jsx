'use client';

import React from 'react';
import {
  Plus,
  MessageSquare,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  BookOpen,
  Activity,
  X,
} from 'lucide-react';

export default function Sidebar({
  conversations = [],
  currentChatId = null,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  isOpen = true,
  onToggle,
  isMobile = false,
  systemStatus = null,
}) {
  return (
    <>
      {/* Mobile Backdrop */}
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40 lg:hidden transition-opacity"
          onClick={onToggle}
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed lg:static top-0 bottom-0 left-0 z-50 flex flex-col bg-[#f7f7f8] border-r border-surface-border transition-all duration-300 ease-in-out ${
          isMobile
            ? isOpen
              ? 'translate-x-0 w-72 shadow-xl'
              : '-translate-x-full w-72'
            : isOpen
            ? 'w-64'
            : 'w-16'
        }`}
      >
        {/* Top Header: Logo & Name */}
        <div className="flex items-center justify-between p-4 border-b border-surface-border/60">
          <div className="flex items-center gap-2.5 overflow-hidden">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-brand-600 text-white shadow-sm flex-shrink-0">
              <BookOpen className="w-4 h-4" />
            </div>
            {(isOpen || isMobile) && (
              <div className="flex flex-col min-w-0">
                <span className="font-semibold text-charcoal-900 text-base tracking-tight truncate">
                  Brainmate
                </span>
                <span className="text-[11px] text-charcoal-500 font-medium truncate">
                  AI Research Assistant
                </span>
              </div>
            )}
          </div>

          {/* Desktop Toggle Button */}
          {!isMobile && (
            <button
              onClick={onToggle}
              className="p-1.5 text-charcoal-500 hover:text-charcoal-800 hover:bg-black/5 rounded-md transition-colors"
              title={isOpen ? 'Collapse sidebar' : 'Expand sidebar'}
              aria-label={isOpen ? 'Collapse sidebar' : 'Expand sidebar'}
            >
              {isOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            </button>
          )}

          {/* Mobile Close Button */}
          {isMobile && (
            <button
              onClick={onToggle}
              className="p-1.5 text-charcoal-500 hover:text-charcoal-800 hover:bg-black/5 rounded-md transition-colors"
              aria-label="Close menu"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* New Chat Button */}
        <div className="p-3">
          <button
            onClick={onNewChat}
            className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl border border-surface-border/80 bg-white hover:bg-gray-50 text-charcoal-800 text-sm font-medium shadow-sm transition-all duration-150 hover:border-brand-300 hover:shadow ${
              !isOpen && !isMobile ? 'justify-center px-0' : ''
            }`}
            title="Start a new research chat"
          >
            <Plus className="w-4 h-4 text-brand-600 flex-shrink-0" />
            {(isOpen || isMobile) && <span>New Chat</span>}
          </button>
        </div>

        {/* Chat History Section */}
        <div className="flex-1 overflow-y-auto px-2 py-1 space-y-1">
          {(isOpen || isMobile) && (
            <div className="px-2 pt-2 pb-1 text-[11px] font-semibold text-charcoal-400 uppercase tracking-wider">
              Recent Chats
            </div>
          )}

          {conversations.length === 0 ? (
            (isOpen || isMobile) && (
              <div className="px-3 py-6 text-center text-xs text-charcoal-400">
                No previous conversations yet.
              </div>
            )
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === currentChatId;
              return (
                <div
                  key={conv.id}
                  className={`group relative flex items-center justify-between rounded-lg px-2.5 py-2 text-sm cursor-pointer transition-colors ${
                    isActive
                      ? 'bg-black/5 text-charcoal-900 font-medium'
                      : 'text-charcoal-600 hover:bg-black/[0.03] hover:text-charcoal-900'
                  } ${!isOpen && !isMobile ? 'justify-center' : ''}`}
                  onClick={() => onSelectChat(conv.id)}
                  title={conv.title || 'Conversation'}
                >
                  <div className="flex items-center gap-2.5 min-w-0 flex-1">
                    <MessageSquare className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-brand-600' : 'text-charcoal-400'}`} />
                    {(isOpen || isMobile) && (
                      <span className="truncate text-xs leading-normal">
                        {conv.title || 'New Research Chat'}
                      </span>
                    )}
                  </div>

                  {/* Delete Conversation Button */}
                  {(isOpen || isMobile) && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteChat(conv.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 text-charcoal-400 hover:text-red-600 hover:bg-red-50 rounded transition-all ml-1 flex-shrink-0"
                      title="Delete conversation"
                      aria-label="Delete conversation"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer: System Status */}
        {(isOpen || isMobile) && (
          <div className="p-3 border-t border-surface-border/60 bg-[#f7f7f8] text-xs">
            <div className="flex items-center gap-2 text-charcoal-500">
              <span
                className={`w-2 h-2 rounded-full ${
                  systemStatus?.ml_service?.status === 'healthy' ? 'bg-emerald-500' : 'bg-amber-400'
                }`}
              />
              <span className="truncate text-[11px]">
                {systemStatus?.ml_service?.status === 'healthy'
                  ? `Model Ready (${systemStatus.ml_service.model?.category_count || 6} arXiv categories)`
                  : 'Backend / Model Offline'}
              </span>
            </div>
          </div>
        )}
      </aside>
    </>
  );
}

