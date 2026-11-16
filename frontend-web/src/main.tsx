import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import '@fontsource/geist-sans/400.css';
import '@fontsource/geist-sans/600.css';
import '@fontsource/geist-mono/400.css';
import '@fontsource/geist-mono/500.css';
import { App } from './App';
import './index.css';

if (typeof window !== 'undefined') {
  const params = new URLSearchParams(window.location.search);
  const tab = params.get('tab');
  if (tab) {
    const tabRoutes: Record<string, string> = {
      landing: '/',
      overview: '/overview',
      live_check: '/live-check',
      transactions: '/transactions',
      cases: '/cases',
      attack_lab: '/attack-lab',
      models: '/models',
      monitoring: '/monitoring',
      batch: '/batch',
      api_docs: '/api-docs',
      how_it_works: '/how-it-works',
      design: '/design',
    };
    const targetRoute = tabRoutes[tab] || `/${tab}`;
    params.delete('tab');
    const rest = params.toString() ? `?${params.toString()}` : '';
    window.history.replaceState({}, '', `${targetRoute}${rest}`);
  }
}

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, 
      gcTime: 15 * 60 * 1000, 
      refetchOnWindowFocus: false, 
      retry: 1,
    },
  },
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>
);

