import { useState, useEffect, useLayoutEffect, lazy, Suspense, startTransition } from 'react';
import { Routes, Route, useNavigate, useLocation, Navigate } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import { Navbar } from './components/Navbar';
import { ToastProvider } from './components/ui/Toast';
import { CommandPalette } from './components/ui/CommandPalette';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { PageLoadingSkeleton, HeroLoadingSkeleton } from './components/ui/LoadingSkeleton';
import { PageTransition } from './components/layout/PageTransition';
import { NotFoundPage } from './pages/NotFoundPage';
import { ChannelProvider } from './context/ChannelContext';
import { AboutDialog } from './components/AboutDialog';
import { IntroAnimationOverlay } from './components/landing/IntroAnimationOverlay';
import { prefetchRoute } from './lib/prefetch';

// Lazy-loaded Core Pages for Modular Tree-shaking & Sub-200KB Initial Gzip Bundle
const LandingPage = lazy(() => import('./pages/LandingPage').then((m) => ({ default: m.LandingPage })));
const OverviewPage = lazy(() => import('./pages/OverviewPage').then((m) => ({ default: m.OverviewPage })));
const LiveCheckPage = lazy(() => import('./pages/LiveCheckPage').then((m) => ({ default: m.LiveCheckPage })));
const TransactionsPage = lazy(() => import('./pages/TransactionsPage').then((m) => ({ default: m.TransactionsPage })));
const CasesPage = lazy(() => import('./pages/CasesPage').then((m) => ({ default: m.CasesPage })));
const EntityViewPage = lazy(() => import('./pages/EntityViewPage').then((m) => ({ default: m.EntityViewPage })));
const AttackLabPage = lazy(() => import('./pages/AttackLabPage').then((m) => ({ default: m.AttackLabPage })));
const ModelsPage = lazy(() => import('./pages/ModelsPage').then((m) => ({ default: m.ModelsPage })));
const MonitoringPage = lazy(() => import('./pages/MonitoringPage').then((m) => ({ default: m.MonitoringPage })));
const BatchScoringPage = lazy(() => import('./pages/BatchScoringPage').then((m) => ({ default: m.BatchScoringPage })));
const ApiDocsPage = lazy(() => import('./pages/ApiDocsPage').then((m) => ({ default: m.ApiDocsPage })));
const DesignShowcase = import.meta.env.DEV
  ? lazy(() => import('./pages/DesignShowcase').then((m) => ({ default: m.DesignShowcase })))
  : () => null;

const TAB_ROUTES: Record<string, string> = {
  landing: '/',
  overview: '/overview',
  live_check: '/live-check',
  transactions: '/transactions',
  cases: '/cases',
  attack_lab: '/attack-lab',
  models: '/models',
  entity: '/entity',
  monitoring: '/monitoring',
  batch: '/batch',
  api_docs: '/api-docs',
  how_it_works: '/how-it-works',
  design: '/design',
};

function getActiveTab(pathname: string): string {
  const p = pathname.replace(/^\//, '').toLowerCase();
  if (!p) return 'landing';
  if (p === 'live-check' || p === 'live_check') return 'live_check';
  if (p === 'attack-lab' || p === 'attack_lab') return 'attack_lab';
  if (p === 'api-docs' || p === 'api_docs') return 'api_docs';
  if (p === 'how-it-works' || p === 'how_it_works') return 'how_it_works';
  return p;
}

export function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('fraudshield-theme');
      if (saved) return saved === 'dark';
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  const [commandOpen, setCommandOpen] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const currentTab = getActiveTab(location.pathname);
  const [showIntro, setShowIntro] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return false;
    return !sessionStorage.getItem('fraudshield-intro-seen');
  });

  const handleIntroComplete = () => {
    sessionStorage.setItem('fraudshield-intro-seen', 'true');
    setShowIntro(false);
  };

  useEffect(() => {
    const handleReplay = () => setShowIntro(true);
    window.addEventListener('fraudshield:replay-intro', handleReplay);
    return () => window.removeEventListener('fraudshield:replay-intro', handleReplay);
  }, []);

  useLayoutEffect(() => {
    if (isDark) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('fraudshield-theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('fraudshield-theme', 'light');
    }
  }, [isDark]);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e: MediaQueryListEvent) => {
      const saved = localStorage.getItem('fraudshield-theme');
      if (!saved) {
        setIsDark(e.matches);
      }
    };
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);


  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const tabParam = params.get('tab');
    if (tabParam) {
      const targetRoute = TAB_ROUTES[tabParam] || `/${tabParam}`;
      params.delete('tab');
      const restSearch = params.toString() ? `?${params.toString()}` : '';
      navigate(`${targetRoute}${restSearch}`, { replace: true });
    }
  }, [location.search, navigate]);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [location.pathname]);

  useEffect(() => {
    const preloadCoreRoutes = () => {
      ['overview', 'live_check', 'transactions', 'cases', 'models'].forEach((k) => {
        prefetchRoute(k);
      });
    };

    if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
      const handle = (window as unknown as { requestIdleCallback: (cb: () => void) => number }).requestIdleCallback(preloadCoreRoutes);
      return () => (window as unknown as { cancelIdleCallback: (id: number) => void }).cancelIdleCallback(handle);
    } else {
      const timer = setTimeout(preloadCoreRoutes, 800);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleTabChange = (tabId: string, searchParams?: string) => {
    const route = TAB_ROUTES[tabId] || `/${tabId}`;
    const target = searchParams ? `${route}?${searchParams}` : route;
    startTransition(() => {
      navigate(target);
    });
  };

  return (
    <ChannelProvider>
      <ToastProvider>
        { }
        <IntroAnimationOverlay
          isOpen={showIntro}
          onComplete={handleIntroComplete}
        />
        <div className="min-h-screen bg-background text-text transition-colors duration-200 flex flex-col font-sans">
          { }
          <a
            href="#main-content"
            className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-50 focus:px-4 focus:py-2 focus:bg-accent focus:text-white focus:font-semibold focus:rounded-md focus:shadow-lg focus:outline-none focus:ring-2 focus:ring-accent"
          >
            Skip to content
          </a>

          <Navbar
            currentTab={currentTab}
            onTabChange={handleTabChange}
            isDark={isDark}
            onToggleDark={() => setIsDark(!isDark)}
            onOpenCommandPalette={() => setCommandOpen(true)}
            onOpenAbout={() => setAboutOpen(true)}
          />

          <CommandPalette
            open={commandOpen}
            onOpenChange={setCommandOpen}
            onNavigate={handleTabChange}
            isDark={isDark}
            onToggleDark={() => setIsDark(!isDark)}
          />

          <AboutDialog
            open={aboutOpen}
            onOpenChange={setAboutOpen}
            onNavigate={handleTabChange}
          />

          <main id="main-content" tabIndex={-1} className="max-w-[1200px] mx-auto px-4 sm:px-6 py-6 flex-1 w-full outline-none">
            <ErrorBoundary fallbackTitle="FraudShield Ledger Error">
              <Suspense fallback={location.pathname === '/' ? <HeroLoadingSkeleton /> : <PageLoadingSkeleton />}>
                <AnimatePresence mode="wait" initial={false}>
                  <PageTransition key={location.pathname}>
                    <Routes location={location}>
                      { }
                      <Route path="/" element={<LandingPage onNavigate={handleTabChange} isDark={isDark} />} />

                      { }
                      <Route path="/overview" element={<OverviewPage onNavigate={handleTabChange} isDark={isDark} />} />

                      { }
                      <Route path="/live-check" element={<LiveCheckPage />} />
                      <Route path="/live_check" element={<Navigate to="/live-check" replace />} />

                      { }
                      <Route path="/transactions" element={<TransactionsPage />} />

                      { }
                      <Route path="/cases" element={<CasesPage />} />

                      { }
                      <Route path="/entity" element={<EntityViewPage isDark={isDark} />} />

                      { }
                      <Route path="/attack-lab" element={<AttackLabPage />} />
                      <Route path="/attack_lab" element={<Navigate to="/attack-lab" replace />} />

                      { }
                      <Route path="/models" element={<ModelsPage />} />

                      { }
                      <Route path="/monitoring" element={<MonitoringPage />} />

                      { }
                      <Route path="/batch" element={<BatchScoringPage />} />
                      <Route path="/batch-scoring" element={<Navigate to="/batch" replace />} />

                      { }
                      <Route path="/api-docs" element={<ApiDocsPage />} />
                      <Route path="/api_docs" element={<Navigate to="/api-docs" replace />} />

                      { }
                      <Route path="/how-it-works" element={<Navigate to="/" replace />} />
                      <Route path="/how_it_works" element={<Navigate to="/" replace />} />

                      { }
                      {import.meta.env.DEV && (
                        <Route
                          path="/design"
                          element={
                            <DesignShowcase
                              isDark={isDark}
                              onOpenCommandPalette={() => setCommandOpen(true)}
                            />
                          }
                        />
                      )}

                      { }
                      <Route path="*" element={<NotFoundPage />} />
                    </Routes>
                  </PageTransition>
                </AnimatePresence>
              </Suspense>
            </ErrorBoundary>
          </main>

          {}
          {currentTab !== 'overview' && (
            <footer className="w-full border-t border-border py-3.5 px-4 sm:px-6 bg-surface/40 text-xs text-muted transition-colors">
              <div className="max-w-[1200px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-center sm:text-left">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                  <span>
                    Data provenance: Public benchmarks &amp; simulated retail payment streams (PaySim, IEEE-CIS, Synthetic UPI) on held-out test splits.
                  </span>
                </div>
                <div className="text-xs text-muted shrink-0">
                  Simulated benchmark replay
                </div>
              </div>
            </footer>
          )}
        </div>
      </ToastProvider>
    </ChannelProvider>
  );
}
