import React, { createContext, useContext, useState, useEffect } from 'react';

export interface ChannelMeta {
  id: string;
  name: string;
  dataSource: string;
  citation: string;
  badge: string;
}

export const CHANNEL_REGISTRY: Record<string, ChannelMeta> = {
  ALL: {
    id: 'ALL',
    name: 'All Channels',
    dataSource: 'Cross-Benchmark Composite Stream',
    citation: 'Aggregated evaluation across Card, Wallets, Account Opening & UPI',
    badge: 'Composite',
  },
  upi: {
    id: 'upi',
    name: 'UPI India',
    dataSource: 'Synthetic Indian UPI Transaction Stream',
    citation: 'Calibrated Behavioral Simulator (50,000 txns, 2.1% fraud)',
    badge: 'UPI',
  },
  card: {
    id: 'card',
    name: 'Card Payments',
    dataSource: 'European Cardholders Benchmark (ULB)',
    citation: 'Dal Pozzolo et al., 2015 (284,807 txns, PCA features V1-V28)',
    badge: 'Card',
  },
  transfer_wallet: {
    id: 'transfer_wallet',
    name: 'Online Transfers & Wallets',
    dataSource: 'PaySim Synthetic Mobile Money',
    citation: 'Lopez-Rojas et al., 2016 (6.36M records, multi-step liquidation)',
    badge: 'Wallet',
  },
  account: {
    id: 'account',
    name: 'Bank Account Opening',
    dataSource: 'Bank Account Fraud (BAF Benchmark)',
    citation: 'Jesus et al., NeurIPS 2022 (1.0M applications, fair lending)',
    badge: 'Account',
  },
};

interface ChannelContextType {
  channel: string;
  setChannel: (channelId: string) => void;
  channelMeta: ChannelMeta;
  allChannels: ChannelMeta[];
}

const ChannelContext = createContext<ChannelContextType | undefined>(undefined);

export const ChannelProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const normalizeChannel = (id: string | null): string => {
    if (!id) return 'ALL';
    const lower = id.toLowerCase();
    if (lower === 'all') return 'ALL';
    if (lower === 'transfers' || lower === 'transfer_wallet') return 'transfer_wallet';
    if (CHANNEL_REGISTRY[id]) return id;
    if (CHANNEL_REGISTRY[lower]) return lower;
    return 'ALL';
  };

  const [channel, setChannelState] = useState<string>(() => {
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search);
      const urlChannel = urlParams.get('channel');
      if (urlChannel) {
        return normalizeChannel(urlChannel);
      }
      const saved = sessionStorage.getItem('fraudshield_active_channel');
      if (saved) {
        return normalizeChannel(saved);
      }
    }
    return 'ALL';
  });

  const setChannel = (newChannel: string) => {
    const normalized = normalizeChannel(newChannel);
    setChannelState(normalized);
    sessionStorage.setItem('fraudshield_active_channel', normalized);
  };

  useEffect(() => {
    sessionStorage.setItem('fraudshield_active_channel', channel);
  }, [channel]);

  const channelMeta = CHANNEL_REGISTRY[channel] || CHANNEL_REGISTRY.ALL;
  const allChannels = Object.values(CHANNEL_REGISTRY);

  return (
    <ChannelContext.Provider value={{ channel, setChannel, channelMeta, allChannels }}>
      {children}
    </ChannelContext.Provider>
  );
};

export const useChannel = (): ChannelContextType => {
  const context = useContext(ChannelContext);
  if (!context) {
    throw new Error('useChannel must be used within a ChannelProvider');
  }
  return context;
};
