import React, { useEffect, useState } from 'react';
import { Command } from 'cmdk';
import {
  Search,
  Home,
  LayoutDashboard,
  Sparkles,
  ReceiptText,
  Briefcase,
  Network,
  FlaskConical,
  Cpu,
  Activity,
  Layers,
  FileCode2,
  Sun,
  Moon,
  ArrowRight,
  User,
} from 'lucide-react';

export interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (tab: string, search?: string) => void;
  isDark: boolean;
  onToggleDark: () => void;
}

const KNOWN_ACCOUNTS = [
  { id: 'C716083600', label: 'C716083600', desc: 'High-volume mule destination hub (PaySim)', tag: 'Mule Hub', risk: 'high' },
  { id: 'C1231006815', label: 'C1231006815', desc: 'High-velocity rapid balance depletion origin', tag: 'High Velocity', risk: 'medium' },
  { id: 'C1666544295', label: 'C1666544295', desc: 'Split-payment structuring cluster endpoint', tag: 'Structuring', risk: 'high' },
  { id: 'C1305486145', label: 'C1305486145', desc: 'Repeated off-hours balance drain account', tag: 'Drip Drain', risk: 'high' },
  { id: 'M473035779', label: 'M473035779', desc: 'Flagged shell merchant endpoint', tag: 'Merchant Shell', risk: 'medium' },
  { id: 'acc_994102', label: 'acc_994102', desc: 'Card channel account — cross-border spike', tag: 'Card Mule', risk: 'high' },
  { id: 'upi_mule_882', label: 'upi_mule_882', desc: 'UPI channel account — cyclic velocity ring', tag: 'UPI Ring', risk: 'high' },
];

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  open,
  onOpenChange,
  onNavigate,
  isDark,
  onToggleDark,
}) => {
  const [search, setSearch] = useState('');

  // Global shortcut listener for Ctrl+K / Cmd+K and Esc
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === 'k' || e.code === 'KeyK' || e.key === 'KeyK')) {
        e.preventDefault();
        onOpenChange(!open);
      }
      if (e.key === 'Escape' && open) {
        e.preventDefault();
        onOpenChange(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [open, onOpenChange]);

  const handleSelect = (action: () => void) => {
    action();
    onOpenChange(false);
    setSearch('');
  };

  if (!open) return null;

  const trimmedQuery = search.trim();
  const isLikelyAccountId = trimmedQuery.length >= 3;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[12vh] px-4">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-200"
        onClick={() => onOpenChange(false)}
      />

      {/* CMDK Container */}
      <Command
        className="relative z-50 w-full max-w-xl rounded-lg border border-border bg-surface shadow-overlay overflow-hidden text-text duration-200 transition-[transform,opacity] font-sans"
        loop
      >
        {/* Search Input Bar */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-border">
          <Search className="w-4 h-4 text-[#52606D] dark:text-[#8A96A3] shrink-0" strokeWidth={1.5} />
          <Command.Input
            value={search}
            onValueChange={setSearch}
            autoFocus
            placeholder="Search pages, commands, or account IDs..."
            className="w-full bg-transparent text-xs text-[#0F172A] dark:text-[#E8EDF2] placeholder:text-[#52606D] dark:placeholder:text-[#8A96A3] focus:outline-none"
          />
          <kbd className="hidden sm:inline-flex items-center gap-0.5 text-xs font-mono px-1.5 py-0.5 rounded bg-slate-100 dark:bg-[#161F28] border border-[#E3E8EE] dark:border-[#1E2630] text-[#52606D] dark:text-[#8A96A3]">
            ESC
          </kbd>
        </div>

        {/* Command List */}
        <Command.List className="max-h-80 overflow-y-auto p-2 text-xs">
          <Command.Empty className="py-8 text-center text-xs text-[#52606D] dark:text-[#8A96A3]">
            No matching results found.
          </Command.Empty>

          {/* Dynamic Search by Query if user enters an Account ID */}
          {isLikelyAccountId && (
            <Command.Group
              heading="Account Deep-Dive"
              className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:text-[#52606D] dark:[&_[cmdk-group-heading]]:text-[#8A96A3] [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 normal-case"
            >
              <Command.Item
                value={`inspect-entity-${trimmedQuery}`}
                onSelect={() => handleSelect(() => onNavigate('entity', `account=${encodeURIComponent(trimmedQuery)}`))}
                className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
              >
                <div className="flex items-center gap-2.5">
                  <Network className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                  <div>
                    <span className="font-medium text-xs">Inspect account <code className="font-mono text-emerald-600 dark:text-[#2DD4A0]">{trimmedQuery}</code> in Entity Graph</span>
                    <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Map multi-hop counterparty network and mule rings</p>
                  </div>
                </div>
                <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
              </Command.Item>

              <Command.Item
                value={`filter-tx-${trimmedQuery}`}
                onSelect={() => handleSelect(() => onNavigate('transactions', `search=${encodeURIComponent(trimmedQuery)}`))}
                className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
              >
                <div className="flex items-center gap-2.5">
                  <ReceiptText className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                  <div>
                    <span className="font-medium text-xs">Filter Ledger for <code className="font-mono text-emerald-600 dark:text-[#2DD4A0]">{trimmedQuery}</code></span>
                    <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Query historical pre-authorization decisions</p>
                  </div>
                </div>
                <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
              </Command.Item>
            </Command.Group>
          )}

          {/* Group 1: Core Pages */}
          <Command.Group
            heading="Navigation Pages"
            className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:text-[#52606D] dark:[&_[cmdk-group-heading]]:text-[#8A96A3] [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 normal-case"
          >
            <Command.Item
              value="overview dashboard metrics kpi"
              onSelect={() => handleSelect(() => onNavigate('overview'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <LayoutDashboard className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Overview</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Protected volume and risk distribution</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="live check scoring pre-auth inference"
              onSelect={() => handleSelect(() => onNavigate('live_check'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Sparkles className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Live check</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Pre-auth transaction scoring and explainability</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="transactions ledger query audit history"
              onSelect={() => handleSelect(() => onNavigate('transactions'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <ReceiptText className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Transactions</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Historical ledger and filter queries</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="cases alerts queue triage investigation"
              onSelect={() => handleSelect(() => onNavigate('cases'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Briefcase className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Cases</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Investigator alert queue and keyboard triage</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="models champion registry architecture metrics"
              onSelect={() => handleSelect(() => onNavigate('models'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Cpu className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Models</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Champion architectures and benchmark metrics</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="landing home the gate hero 3d"
              onSelect={() => handleSelect(() => onNavigate('landing'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Home className="w-4 h-4 text-emerald-600 dark:text-[#2DD4A0]" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Landing Page</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">The Gate decision visualizer and product hero</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>
          </Command.Group>

          {/* Group 2: Secondary Tools */}
          <Command.Group
            heading="Extended Views"
            className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:text-[#52606D] dark:[&_[cmdk-group-heading]]:text-[#8A96A3] [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 mt-2 normal-case"
          >
            <Command.Item
              value="entity view graph mule hub counterparty"
              onSelect={() => handleSelect(() => onNavigate('entity'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Network className="w-4 h-4" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Entity View</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Counterparty graph mapping & mule hub rings</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="attack lab adversarial evasion simulation"
              onSelect={() => handleSelect(() => onNavigate('attack_lab'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <FlaskConical className="w-4 h-4" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Attack Lab</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Interactive evasion testing &amp; split-transfer simulation</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="monitoring psi concept drift stability"
              onSelect={() => handleSelect(() => onNavigate('monitoring'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Activity className="w-4 h-4" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Monitoring</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">PSI score stability and feature drift tracking</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="batch scoring bulk vectorized csv"
              onSelect={() => handleSelect(() => onNavigate('batch'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <Layers className="w-4 h-4" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">Batch Scoring</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Vectorized bulk evaluation of transaction files</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>

            <Command.Item
              value="api docs rest curl schema latency"
              onSelect={() => handleSelect(() => onNavigate('api_docs'))}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                <FileCode2 className="w-4 h-4" strokeWidth={1.5} />
                <div>
                  <span className="font-medium text-xs">API Docs</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">REST endpoint specifications & cURL snippets</p>
                </div>
              </div>
              <ArrowRight className="w-3.5 h-3.5 opacity-40" strokeWidth={1.5} />
            </Command.Item>
          </Command.Group>

          {/* Group 3: Known Accounts & Entities */}
          <Command.Group
            heading="Known Account Topologies"
            className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:text-[#52606D] dark:[&_[cmdk-group-heading]]:text-[#8A96A3] [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 mt-2 normal-case"
          >
            {KNOWN_ACCOUNTS.map((acc) => (
              <Command.Item
                key={acc.id}
                value={`${acc.id} ${acc.label} ${acc.desc} ${acc.tag}`}
                onSelect={() => handleSelect(() => onNavigate('entity', `account=${acc.id}`))}
                className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
              >
                <div className="flex items-center gap-2.5">
                  <User className={`w-4 h-4 ${acc.risk === 'high' ? 'text-rose-500' : 'text-amber-500'}`} strokeWidth={1.5} />
                  <div>
                    <span className="font-medium text-xs font-mono">{acc.label}</span>
                    <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">{acc.desc}</p>
                  </div>
                </div>
                <span className={`text-xs font-mono px-1.5 py-0.5 rounded border ${
                  acc.risk === 'high'
                    ? 'bg-rose-500/10 text-rose-600 dark:text-rose-400 border-rose-500/20'
                    : 'bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20'
                }`}>
                  {acc.tag}
                </span>
              </Command.Item>
            ))}
          </Command.Group>

          {/* Group 4: Quick System Actions */}
          <Command.Group
            heading="System Actions"
            className="[&_[cmdk-group-heading]]:text-xs [&_[cmdk-group-heading]]:font-semibold [&_[cmdk-group-heading]]:text-[#52606D] dark:[&_[cmdk-group-heading]]:text-[#8A96A3] [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1 mt-2 normal-case"
          >
            <Command.Item
              value="theme toggle dark light mode switch"
              onSelect={() => handleSelect(onToggleDark)}
              className="flex items-center justify-between px-2.5 py-2 rounded-md cursor-pointer hover:bg-slate-100 dark:hover:bg-[#161F28] aria-selected:bg-emerald-500/10 dark:aria-selected:bg-emerald-500/15 aria-selected:text-emerald-700 dark:aria-selected:text-[#2DD4A0] transition-colors"
            >
              <div className="flex items-center gap-2.5">
                {isDark ? (
                  <Sun className="w-4 h-4 text-amber-400" strokeWidth={1.5} />
                ) : (
                  <Moon className="w-4 h-4 text-slate-700" strokeWidth={1.5} />
                )}
                <div>
                  <span className="font-medium text-xs">Switch to {isDark ? 'Light' : 'Dark'} Mode</span>
                  <p className="text-xs text-[#52606D] dark:text-[#8A96A3]">Toggle security control room theme</p>
                </div>
              </div>
              <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-slate-100 dark:bg-[#161F28] border border-[#E3E8EE] dark:border-[#1E2630] text-[#52606D] dark:text-[#8A96A3]">
                Toggle
              </span>
            </Command.Item>
          </Command.Group>
        </Command.List>

        {/* Footer Bar */}
        <div className="px-4 py-2 border-t border-[#E3E8EE] dark:border-[#1E2630] bg-slate-50/50 dark:bg-[#0B0F14]/50 flex items-center justify-between text-xs text-[#52606D] dark:text-[#8A96A3] font-mono">
          <div className="flex items-center gap-2">
            <span>↑↓ Navigate</span>
            <span>↵ Select</span>
            <span>ESC Close</span>
          </div>
          <span className="text-emerald-600 dark:text-[#2DD4A0]">Ctrl+K</span>
        </div>
      </Command>
    </div>
  );
};
