const routePreloaders: Record<string, () => Promise<unknown>> = {
  landing: () => import('../pages/LandingPage'),
  overview: () => import('../pages/OverviewPage'),
  live_check: () => import('../pages/LiveCheckPage'),
  transactions: () => import('../pages/TransactionsPage'),
  cases: () => import('../pages/CasesPage'),
  entity: () => import('../pages/EntityViewPage'),
  attack_lab: () => import('../pages/AttackLabPage'),
  models: () => import('../pages/ModelsPage'),
  monitoring: () => import('../pages/MonitoringPage'),
  batch: () => import('../pages/BatchScoringPage'),
  api_docs: () => import('../pages/ApiDocsPage'),
};

export function prefetchRoute(tabKey: string) {
  const preload = routePreloaders[tabKey];
  if (preload) {
    preload().catch(() => {});
  }
}

