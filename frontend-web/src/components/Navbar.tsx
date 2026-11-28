import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MagnifyingGlass,
  Sun,
  Moon,
  CaretDown,
  List,
  X,
  Check,
  Info,
} from '@phosphor-icons/react';
import { Popover } from './ui/Popover';
import { Tooltip } from './ui/Tooltip';
import { useChannel } from '../context/ChannelContext';
import { prefetchRoute } from '../lib/prefetch';

export interface NavbarProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  isDark: boolean;
  onToggleDark: () => void;
  onOpenCommandPalette: () => void;
  onOpenAbout: () => void;
}

interface NavLinkItemProps {
  id: string;
  label: string;
  active: boolean;
  onClick: () => void;
}

export const NavLinkItem: React.FC<NavLinkItemProps> = ({ id, label, active, onClick }) => {
  return (
    <button
      key={id}
      onClick={onClick}
      onMouseEnter={() => prefetchRoute(id)}
      onFocus={() => prefetchRoute(id)}
      aria-current={active ? 'page' : undefined}
      className="relative px-3 py-1.5 rounded-md text-xs font-medium text-muted hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 transition-colors select-none"
    >
      {active && (
        <motion.div
          layoutId="navbar-active-pill"
          className="absolute inset-0 rounded-md bg-surface-subtle border border-border"
          transition={{ type: 'spring', stiffness: 450, damping: 35 }}
        />
      )}
      <span className={`relative z-10 ${active ? 'font-semibold text-text' : ''}`}>
        {label}
      </span>
    </button>
  );
};

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  onTabChange,
  isDark,
  onToggleDark,
  onOpenCommandPalette,
  onOpenAbout,
}) => {
  const { channel, setChannel, channelMeta, allChannels } = useChannel();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  // Transparent over hero, frosted after 8px scroll or on app pages
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 8);
    };
    handleScroll();
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile drawer on Escape key
  useEffect(() => {
    if (!mobileOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileOpen(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileOpen]);

  const handleMoreKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    const items = e.currentTarget.querySelectorAll<HTMLButtonElement>('button');
    const focused = document.activeElement as HTMLButtonElement;
    const index = Array.from(items).indexOf(focused);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = index < items.length - 1 ? items[index + 1] : items[0];
      next?.focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = index > 0 ? items[index - 1] : items[items.length - 1];
      prev?.focus();
    }
  };

  // Close mobile sheet on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && mobileOpen) {
        setMobileOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mobileOpen]);

  // Primary 5 navigation routes
  const primaryTabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'live_check', label: 'Live check' },
    { id: 'transactions', label: 'Transactions' },
    { id: 'cases', label: 'Cases' },
    { id: 'models', label: 'Models' },
  ];

  // Secondary grouped items for "More" menu
  const analysisTabs = [
    { id: 'attack_lab', label: 'Attack Lab' },
    { id: 'entity', label: 'Entity view' },
    { id: 'monitoring', label: 'Monitoring' },
  ];

  const toolTabs = [
    { id: 'batch', label: 'Batch scoring' },
    { id: 'api_docs', label: 'API docs' },
    ...(import.meta.env.DEV ? [{ id: 'design', label: 'Design system' }] : []),
  ];

  const isMoreActive =
    analysisTabs.some((t) => t.id === currentTab) ||
    toolTabs.some((t) => t.id === currentTab);

  const isLanding = currentTab === 'landing';
  const isFrosted = !isLanding || isScrolled;

  const handleNavigate = (tabId: string) => {
    onTabChange(tabId);
    setMobileOpen(false);
  };

  return (
    <>
      <header
        className={`sticky top-0 z-40 w-full h-14 transition-colors duration-200 ${
          isFrosted
            ? 'bg-surface/85 backdrop-blur-md border-b border-border shadow-xs'
            : 'bg-transparent border-b border-transparent'
        }`}
      >
        <div className="max-w-[1200px] mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-4">
          {/* ===================================================================
              ZONE 1 (LEFT): LOGO MARK + "FraudShield" (No LIVE badge, No version)
              =================================================================== */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => handleNavigate('landing')}
              className="flex items-center gap-2.5 text-left focus:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-md group shrink-0"
              title="FraudShield Home"
              aria-label="FraudShield Home"
            >
              <div className="w-7 h-7 rounded-md bg-surface border border-border flex items-center justify-center transition-colors group-hover:border-accent">
                <svg className="w-4 h-4" viewBox="0 0 32 32" fill="none">
                  <path
                    d="M16 4.5L25 8.5V15.5C25 21 21.5 25.5 16 27.5C10.5 25.5 7 21 7 15.5V8.5L16 4.5Z"
                    className="fill-surface-subtle stroke-border"
                    strokeWidth="1.2"
                  />
                  <polygon
                    points="16,11 20,14 20,18 16,21 12,18 12,14"
                    className="fill-surface stroke-accent"
                    strokeWidth="1.2"
                  />
                  <circle cx="16" cy="16" r="2.2" className="fill-accent" />
                </svg>
              </div>
              <span className="font-sans font-semibold text-sm tracking-normal text-text">
                FraudShield
              </span>
            </button>
          </div>

          {/* ===================================================================
              ZONE 2 (CENTRE): 5 PRIMARY TABS + COMPACT "MORE" MENU
              =================================================================== */}
          <nav
            aria-label="Main Navigation"
            className="hidden md:flex items-center gap-0.5"
          >
            {primaryTabs.map((tab) => (
              <NavLinkItem
                key={tab.id}
                id={tab.id}
                label={tab.label}
                active={currentTab === tab.id}
                onClick={() => handleNavigate(tab.id)}
              />
            ))}

            {/* "More" Menu: Plain grouped list of labels */}
            <Popover
              align="start"
              trigger={
                <button
                  aria-label="More platform views and tools"
                  aria-expanded={isMoreActive}
                  className="relative flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium text-muted hover:text-text focus:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 transition-colors select-none"
                >
                  {isMoreActive && (
                    <motion.div
                      layoutId="navbar-active-pill"
                      className="absolute inset-0 rounded-md bg-surface-subtle border border-border"
                      transition={{ type: 'spring', stiffness: 450, damping: 35 }}
                    />
                  )}
                  <span
                    className={`relative z-10 flex items-center gap-1 ${
                      isMoreActive ? 'font-semibold text-text' : ''
                    }`}
                  >
                    <span>More</span>
                    <CaretDown className="w-3 h-3 text-muted" weight="thin" />
                  </span>
                </button>
              }
            >
              <div
                role="menu"
                onKeyDown={handleMoreKeyDown}
                className="w-52 p-1.5 bg-surface border border-border rounded-lg shadow-xl text-left text-xs font-sans space-y-2"
              >
                <div>
                  <div className="px-2.5 py-1 text-xs font-semibold text-muted">
                    Analysis
                  </div>
                  <div className="space-y-0.5">
                    {analysisTabs.map((tab) => (
                      <button
                        key={tab.id}
                        role="menuitem"
                        onClick={() => handleNavigate(tab.id)}
                        onMouseEnter={() => prefetchRoute(tab.id)}
                        onFocus={() => prefetchRoute(tab.id)}
                        className={`w-full px-2.5 py-1.5 rounded-md text-left transition-colors flex items-center justify-between focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                          currentTab === tab.id
                            ? 'bg-surface-subtle text-text font-semibold'
                            : 'text-muted hover:text-text hover:bg-surface-subtle'
                        }`}
                      >
                        <span>{tab.label}</span>
                        {currentTab === tab.id && (
                          <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="pt-1.5 border-t border-border">
                  <div className="px-2.5 py-1 text-xs font-semibold text-muted">
                    Tools
                  </div>
                  <div className="space-y-0.5">
                    {toolTabs.map((tab) => (
                      <button
                        key={tab.id}
                        role="menuitem"
                        onClick={() => handleNavigate(tab.id)}
                        onMouseEnter={() => prefetchRoute(tab.id)}
                        onFocus={() => prefetchRoute(tab.id)}
                        className={`w-full px-2.5 py-1.5 rounded-md text-left transition-colors flex items-center justify-between focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                          currentTab === tab.id
                            ? 'bg-surface-subtle text-text font-semibold'
                            : 'text-muted hover:text-text hover:bg-surface-subtle'
                        }`}
                      >
                        <span>{tab.label}</span>
                        {currentTab === tab.id && (
                          <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                        )}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="pt-1.5 border-t border-border">
                  <button
                    role="menuitem"
                    onClick={() => {
                      onOpenAbout();
                    }}
                    className="w-full px-2.5 py-1.5 rounded-md text-left transition-colors flex items-center gap-1.5 text-muted hover:text-text hover:bg-surface-subtle focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    <Info className="w-3.5 h-3.5" weight="thin" />
                    <span>About FraudShield</span>
                  </button>
                </div>
              </div>
            </Popover>
          </nav>

          {/* ===================================================================
              ZONE 3 (RIGHT): COMPACT CHANNEL SWITCHER, CTRL+K SEARCH, THEME TOGGLE
              =================================================================== */}
          <div className="flex items-center gap-2">
            {/* Compact Channel Switcher */}
            <Popover
              align="end"
              trigger={
                <button
                  className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-2.5 py-1.5 rounded-md border border-border bg-surface hover:bg-surface-subtle text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent whitespace-nowrap"
                  title="Switch Payment Channel"
                  aria-label={`Current channel: ${channelMeta.name}`}
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                  <span className="text-xs font-medium text-text whitespace-nowrap">
                    <span className="hidden sm:inline">{channelMeta.name}</span>
                    <span className="sm:hidden">{channel === 'all' ? 'All' : channelMeta.name}</span>
                  </span>
                  <CaretDown className="w-3 h-3 text-muted shrink-0" weight="thin" />
                </button>
              }
            >
              <div className="w-72 p-1.5 bg-surface border border-border rounded-lg shadow-xl text-left text-xs font-sans space-y-1">
                <div className="px-2.5 py-1 text-xs font-semibold text-muted border-b border-border pb-1">
                  Active Channel & Provenance
                </div>
                {allChannels.map((item) => {
                  const isSelected = item.id === channel;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setChannel(item.id)}
                      className={`w-full flex items-start justify-between gap-2 p-2 rounded-md text-left transition-colors ${
                        isSelected
                          ? 'bg-surface-subtle text-text font-semibold'
                          : 'text-muted hover:text-text hover:bg-surface-subtle'
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="font-semibold text-xs text-text">{item.name}</div>
                        <div className="text-xs text-muted font-sans mt-0.5">
                          {item.dataSource}
                        </div>
                        <div className="text-xs text-muted/70 font-sans italic mt-0.5">
                          {item.citation}
                        </div>
                      </div>
                      {isSelected && (
                        <Check className="w-4 h-4 text-accent shrink-0 mt-0.5" weight="thin" />
                      )}
                    </button>
                  );
                })}
              </div>
            </Popover>

            {/* Ctrl+K Search Icon-Button with Tooltip */}
            <Tooltip content="Search pages and accounts (Ctrl+K)">
              <button
                onClick={onOpenCommandPalette}
                aria-label="Search pages and accounts (Ctrl+K)"
                className="w-8 h-8 rounded-md border border-border bg-surface hover:bg-surface-subtle text-muted hover:text-text flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                <MagnifyingGlass className="w-4 h-4" weight="thin" />
              </button>
            </Tooltip>

            {/* Theme Toggle Icon-Button with Tooltip */}
            <Tooltip content="Toggle color theme">
              <button
                onClick={onToggleDark}
                aria-label="Toggle color theme"
                className="w-8 h-8 rounded-md border border-border bg-surface hover:bg-surface-subtle text-muted hover:text-text flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
              >
                {isDark ? (
                  <Sun className="w-4 h-4 text-amber-400" weight="thin" />
                ) : (
                  <Moon className="w-4 h-4 text-text" weight="thin" />
                )}
              </button>
            </Tooltip>

            {/* Mobile Sheet Hamburger */}
            <button
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation menu"
              className="md:hidden w-11 h-11 rounded-md border border-border bg-surface hover:bg-surface-subtle text-muted hover:text-text flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            >
              <List className="w-5 h-5" weight="thin" />
            </button>
          </div>
        </div>
      </header>

      {/* =======================================================================
          MOBILE FULL-HEIGHT SHEET (From the Right, No Duplicate Grids)
          ======================================================================= */}
      <AnimatePresence>
        {mobileOpen && (
          <div className="fixed inset-0 z-50 md:hidden">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setMobileOpen(false)}
              className="absolute inset-0 bg-black/50 backdrop-blur-xs"
            />

            {/* Sliding Panel */}
            <motion.aside
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: 'spring', damping: 30, stiffness: 350 }}
              className="absolute inset-y-0 right-0 w-80 max-w-[85vw] bg-surface border-l border-border shadow-2xl p-5 flex flex-col justify-between overflow-y-auto"
            >
              <div className="space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between pb-3 border-b border-border">
                  <div className="flex items-center gap-2">
                    <div className="w-6 h-6 rounded bg-surface border border-border flex items-center justify-center">
                      <div className="w-2 h-2 rounded-full bg-accent" />
                    </div>
                    <span className="font-semibold text-sm text-text">FraudShield</span>
                  </div>
                  <button
                    onClick={() => setMobileOpen(false)}
                    aria-label="Close navigation menu"
                    className="w-11 h-11 rounded-md text-muted hover:text-text flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    <X className="w-5 h-5" weight="thin" />
                  </button>
                </div>

                {/* Primary Nav List */}
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-muted px-2 py-1">
                    Navigation
                  </div>
                  {primaryTabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => handleNavigate(tab.id)}
                      className={`w-full min-h-[44px] px-3 py-2.5 rounded-md text-xs text-left font-medium transition-colors flex items-center justify-between focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                        currentTab === tab.id
                          ? 'bg-surface-subtle text-text font-semibold'
                          : 'text-muted hover:text-text hover:bg-surface-subtle'
                      }`}
                    >
                      <span>{tab.label}</span>
                      {currentTab === tab.id && (
                        <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                      )}
                    </button>
                  ))}
                </div>

                {/* Analysis Group */}
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-muted px-2 py-1">
                    Analysis
                  </div>
                  {analysisTabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => handleNavigate(tab.id)}
                      className={`w-full min-h-[44px] px-3 py-2.5 rounded-md text-xs text-left font-medium transition-colors flex items-center justify-between focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                        currentTab === tab.id
                          ? 'bg-surface-subtle text-text font-semibold'
                          : 'text-muted hover:text-text hover:bg-surface-subtle'
                      }`}
                    >
                      <span>{tab.label}</span>
                      {currentTab === tab.id && (
                        <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                      )}
                    </button>
                  ))}
                </div>

                {/* Tools Group */}
                <div className="space-y-1">
                  <div className="text-xs font-semibold text-muted px-2 py-1">
                    Tools & Information
                  </div>
                  {toolTabs.map((tab) => (
                    <button
                      key={tab.id}
                      onClick={() => handleNavigate(tab.id)}
                      className={`w-full min-h-[44px] px-3 py-2.5 rounded-md text-xs text-left font-medium transition-colors flex items-center justify-between focus:outline-none focus-visible:ring-2 focus-visible:ring-accent ${
                        currentTab === tab.id
                          ? 'bg-surface-subtle text-text font-semibold'
                          : 'text-muted hover:text-text hover:bg-surface-subtle'
                      }`}
                    >
                      <span>{tab.label}</span>
                      {currentTab === tab.id && (
                        <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                      )}
                    </button>
                  ))}
                  <button
                    onClick={() => {
                      setMobileOpen(false);
                      onOpenAbout();
                    }}
                    className="w-full min-h-[44px] px-3 py-2.5 rounded-md text-xs text-left font-medium text-muted hover:text-text hover:bg-surface-subtle flex items-center gap-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
                  >
                    <Info className="w-3.5 h-3.5" weight="thin" />
                    <span>About FraudShield</span>
                  </button>
                </div>
              </div>

              {/* Bottom Channel Indicator */}
              <div className="pt-4 border-t border-border mt-4 text-xs font-sans pb-[calc(1.5rem+env(safe-area-inset-bottom,0px))]">
                <div className="text-muted text-xs mb-1">Active Payment Channel:</div>
                <div className="font-semibold text-text flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-accent" />
                  <span>{channelMeta.name}</span>
                </div>
                <div className="text-xs text-muted mt-0.5">{channelMeta.dataSource}</div>
              </div>
            </motion.aside>
          </div>
        )}
      </AnimatePresence>
    </>
  );
};
