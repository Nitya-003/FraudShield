import React, { useState, useMemo, useEffect, useRef } from 'react';
import { useQuery, useQueryClient, keepPreviousData } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { fetchTransactions } from '../api/client';
import type { TransactionItem } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Select } from '../components/ui/Select';
import { Input } from '../components/ui/Input';
import { Button } from '../components/ui/Button';
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from '../components/ui/Sheet';
import { PageHeader } from '../components/layout/PageHeader';
import { useToast } from '../components/ui/Toast';
import { Skeleton } from '../components/ui/Skeleton';
import {
  Download,
  Copy,
  Network,
  AlertCircle,
  Inbox,
  X,
  ArrowRight,
  ShieldAlert,
  SlidersHorizontal,
} from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  getChannelCurrency,
  getChannelReadableName,
  formatChannelAmount,
} from '../lib/channelUtils';

interface CopyablePartyProps {
  value?: string;
  label: string;
}

const CopyableParty: React.FC<CopyablePartyProps> = ({ value, label }) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  if (!value) return <span className="text-muted text-xs">—</span>;

  const truncated = value.length > 18 ? `${value.slice(0, 7)}…${value.slice(-6)}` : value;

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
    toast({
      title: 'Copied to clipboard',
      description: `${label}: ${value}`,
      variant: 'info',
    });
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="group/party inline-flex items-center gap-1 font-mono text-xs text-text hover:text-accent transition-colors cursor-pointer text-left"
      title={`Click to copy ${label}: ${value}`}
    >
      <span className="truncate max-w-[130px]">{truncated}</span>
      <Copy className="w-3 h-3 opacity-0 group-hover/party:opacity-80 transition-opacity shrink-0 text-muted" />
      {copied && <span className="text-xs text-accent font-sans shrink-0">Copied</span>}
    </button>
  );
};

export const TransactionsPage: React.FC = () => {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [channel, setChannel] = useState('ALL');
  const [band, setBand] = useState('ALL');
  const [searchInput, setSearchInput] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [selectedRowIndex, setSelectedRowIndex] = useState<number>(-1);
  const [detailTx, setDetailTx] = useState<TransactionItem | null>(null);

  // 250ms debounce on search
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearch(searchInput);
      setPage(1);
    }, 250);
    return () => clearTimeout(handler);
  }, [searchInput]);

  // Default sort is newest first (step descending, not score descending)
  const { data, isLoading, isFetching, isError, refetch } = useQuery({
    queryKey: ['transactions', page, pageSize, channel, band, debouncedSearch],
    queryFn: () =>
      fetchTransactions({
        page,
        pageSize,
        channel,
        band,
        search: debouncedSearch.trim() || undefined,
        sortBy: 'step',
        sortOrder: 'desc',
      }),
    placeholderData: keepPreviousData,
    staleTime: 60 * 1000,
  });

  const items = useMemo(() => data?.items || [], [data?.items]);

  // Prefetch next page
  useEffect(() => {
    const hasMore = data ? page * pageSize < data.total : false;
    if (hasMore) {
      queryClient.prefetchQuery({
        queryKey: ['transactions', page + 1, pageSize, channel, band, debouncedSearch],
        queryFn: () =>
          fetchTransactions({
            page: page + 1,
            pageSize,
            channel,
            band,
            search: debouncedSearch.trim() || undefined,
            sortBy: 'step',
            sortOrder: 'desc',
          }),
        staleTime: 60 * 1000,
      });
    }
  }, [data, page, pageSize, channel, band, debouncedSearch, queryClient]);

  // Keyboard row navigation: ArrowUp, ArrowDown, Enter
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const activeEl = document.activeElement;
      if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.tagName === 'SELECT')) {
        return;
      }
      if (detailTx) return;
      if (items.length === 0) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedRowIndex((prev) => Math.min(items.length - 1, prev + 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedRowIndex((prev) => Math.max(0, prev - 1));
      } else if (e.key === 'Enter' && selectedRowIndex >= 0 && selectedRowIndex < items.length) {
        e.preventDefault();
        setDetailTx(items[selectedRowIndex]);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [items, detailTx, selectedRowIndex]);

  // Virtualized row container
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const rowVirtualizer = useVirtualizer({
    count: items.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 44, // 44px rows
    overscan: 8,
  });

  const hasActiveFilters = channel !== 'ALL' || band !== 'ALL' || debouncedSearch.trim().length > 0;

  const handleClearAllFilters = () => {
    setChannel('ALL');
    setBand('ALL');
    setSearchInput('');
    setDebouncedSearch('');
    setPage(1);
    setSelectedRowIndex(-1);
  };

  // Export CSV
  const handleExportCSV = () => {
    if (items.length === 0) {
      toast({
        title: 'Export Failed',
        description: 'No transactions available to export.',
        variant: 'warning',
      });
      return;
    }

    const headers = ['Step', 'Channel', 'Type', 'Sender_ID', 'Recipient_ID', 'Amount', 'Currency', 'Risk_Score', 'Risk_Band', 'Reason'];
    const rows = items.map((item) => [
      item.step,
      getChannelReadableName(item.channel),
      item.type,
      `"${item.nameOrig || ''}"`,
      `"${item.nameDest || ''}"`,
      item.amount,
      getChannelCurrency(item.channel),
      item.risk_score.toFixed(4),
      item.risk_band,
      `"${(item.reason || '').replace(/"/g, '""')}"`,
    ]);

    const csvContent = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `fraudshield_transactions_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: 'CSV Export Generated',
      description: `Successfully exported ${items.length} transactions.`,
      variant: 'success',
    });
  };

  const totalPages = Math.max(1, Math.ceil((data?.total || 0) / pageSize));

  return (
    <div className="space-y-6 pb-[calc(3rem+env(safe-area-inset-bottom,0px))]">
      {/* Consistent Page Header */}
      <PageHeader
        title="Transactions"
        purpose="Multi-channel payment evaluation ledger with calibrated risk bands and local attribution."
        action={
          <Button variant="secondary" size="sm" onClick={handleExportCSV} className="min-h-[36px]">
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </Button>
        }
      />

      {/* Error state with Retry */}
      {isError && (
        <div className="p-4 rounded-xl border border-danger/30 bg-danger/10 text-xs text-danger flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>Failed to load transactions ledger from API.</span>
          </div>
          <Button variant="secondary" size="sm" onClick={() => refetch()} className="min-h-[36px]">
            Retry
          </Button>
        </div>
      )}

      {/* Filter Bar with Custom Select Filters */}
      <div className="space-y-3">
        <div className="p-3.5 rounded-xl border border-border bg-surface flex flex-col md:flex-row items-center justify-between gap-3 shadow-card">
          <div className="flex-1 w-full md:w-auto flex items-center gap-2">
            <Input
              placeholder="Search by counterparty ID, handle, or transaction type..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
            {isFetching && !isLoading && (
              <span className="shrink-0 inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-sans text-accent bg-accent/10 border border-accent/20 animate-pulse">
                <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                Updating
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 w-full md:w-auto shrink-0">
            <Select
              aria-label="Filter by channel"
              className="min-w-[150px]"
              value={channel}
              onChange={(e) => {
                setChannel(e.target.value);
                setPage(1);
                setSelectedRowIndex(-1);
              }}
              options={[
                { value: 'ALL', label: 'All Channels' },
                { value: 'upi', label: 'UPI Instant' },
                { value: 'transfer_wallet', label: 'Wallet Transfer' },
                { value: 'card', label: 'Card Payment' },
                { value: 'account', label: 'Account Opening' },
              ]}
            />

            <Select
              aria-label="Filter by risk band"
              className="min-w-[150px]"
              value={band}
              onChange={(e) => {
                setBand(e.target.value);
                setPage(1);
                setSelectedRowIndex(-1);
              }}
              options={[
                { value: 'ALL', label: 'All Risk Bands' },
                { value: 'Low', label: 'Low (Approve)' },
                { value: 'Medium', label: 'Medium (Step-Up)' },
                { value: 'High', label: 'High (Decline)' },
              ]}
            />
          </div>
        </div>

        {/* Visible Active-Filter Chip Row with Clear All */}
        {hasActiveFilters && (
          <div className="flex flex-wrap items-center gap-2 pt-0.5 text-xs">
            <span className="text-muted flex items-center gap-1">
              <SlidersHorizontal className="w-3 h-3" />
              <span>Active filters:</span>
            </span>

            {channel !== 'ALL' && (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface border border-border text-text font-medium">
                Channel: {getChannelReadableName(channel)}
                <button
                  type="button"
                  onClick={() => {
                    setChannel('ALL');
                    setPage(1);
                  }}
                  className="hover:text-danger ml-0.5"
                  aria-label="Remove channel filter"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            )}

            {band !== 'ALL' && (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface border border-border text-text font-medium">
                Risk Band: {band}
                <button
                  type="button"
                  onClick={() => {
                    setBand('ALL');
                    setPage(1);
                  }}
                  className="hover:text-danger ml-0.5"
                  aria-label="Remove band filter"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            )}

            {debouncedSearch.trim() && (
              <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-surface border border-border text-text font-medium">
                Search: &ldquo;{debouncedSearch.trim()}&rdquo;
                <button
                  type="button"
                  onClick={() => {
                    setSearchInput('');
                    setDebouncedSearch('');
                    setPage(1);
                  }}
                  className="hover:text-danger ml-0.5"
                  aria-label="Remove search filter"
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            )}

            <button
              type="button"
              onClick={handleClearAllFilters}
              className="text-xs text-accent hover:underline font-semibold ml-1 cursor-pointer"
            >
              Clear all
            </button>
          </div>
        )}
      </div>

      {/* Main Ledger Table */}
      <div className="bg-surface border border-border rounded-xl shadow-card overflow-hidden">
        {/* Desktop / Tablet Table */}
        <div
          ref={tableContainerRef}
          tabIndex={0}
          aria-label="Transactions ledger table with keyboard arrow navigation"
          className="hidden md:block overflow-x-auto max-h-[640px] focus:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          <table className="w-full min-w-[960px] text-left text-xs border-collapse">
            {/* Sticky Header with sentence-case 12px muted 600 */}
            <thead className="bg-surface-subtle border-b border-border sticky top-0 z-20">
              <tr className="h-10">
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[90px]">
                  Step / Time
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[130px]">
                  Channel
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[180px]">
                  Parties
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal text-right w-[110px]">
                  Amount
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[130px]">
                  Score
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal w-[100px]">
                  Risk band
                </th>
                <th scope="col" className="py-2.5 px-3.5 font-semibold text-muted text-xs tracking-normal min-w-[220px]">
                  Primary driver
                </th>
              </tr>
            </thead>

            <tbody className="divide-y divide-border font-sans">
              {isLoading && items.length === 0 ? (
                Array.from({ length: 8 }).map((_, rIdx) => (
                  <tr key={`skel-${rIdx}`} className="h-[44px] animate-pulse">
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-12" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-20" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-28" /></td>
                    <td className="py-2.5 px-3.5 text-right"><div className="h-3 bg-border rounded w-16 ml-auto" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-20" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-16" /></td>
                    <td className="py-2.5 px-3.5"><div className="h-3 bg-border rounded w-44" /></td>
                  </tr>
                ))
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-16 text-center text-xs text-muted">
                    <div className="max-w-xs mx-auto space-y-2">
                      <Inbox className="w-8 h-8 mx-auto opacity-40 text-muted" />
                      <div className="font-semibold text-sm text-text">No transactions match your filters</div>
                      <p className="text-muted">Adjust channel, band, or query criteria to view historical records.</p>
                      <Button variant="secondary" size="sm" onClick={handleClearAllFilters} className="mt-2">
                        Reset filters
                      </Button>
                    </div>
                  </td>
                </tr>
              ) : (
                rowVirtualizer.getVirtualItems().map((virtualRow) => {
                  const item = items[virtualRow.index];
                  if (!item) return null;
                  const isKeyboardSelected = selectedRowIndex === virtualRow.index;
                  const scoreClamped = Math.max(0, Math.min(1, item.risk_score));
                  const scorePct = Math.round(scoreClamped * 100);

                  const barColor =
                    item.risk_band === 'High'
                      ? 'bg-danger'
                      : item.risk_band === 'Medium'
                        ? 'bg-warn'
                        : 'bg-accent';

                  return (
                    <tr
                      key={`${item.channel}-${item.step}-${virtualRow.index}`}
                      onClick={() => {
                        setSelectedRowIndex(virtualRow.index);
                        setDetailTx(item);
                      }}
                      className={`h-[44px] cursor-pointer transition-colors duration-120 select-none ${isKeyboardSelected
                          ? 'bg-surface-subtle ring-1 ring-inset ring-accent/40'
                          : 'hover:bg-surface-subtle'
                        }`}
                    >
                      {/* Step / Time */}
                      <td className="py-2.5 px-3.5 font-mono text-xs tabular-nums text-muted">
                        Step {item.step}
                      </td>

                      {/* Channel: Readable name, 14px, normal contrast */}
                      <td className="py-2.5 px-3.5 text-sm font-normal text-text whitespace-nowrap">
                        {getChannelReadableName(item.channel)}
                      </td>

                      {/* Parties: mono, truncated with copy-on-click */}
                      <td className="py-2.5 px-3.5">
                        <div className="flex items-center gap-1.5 whitespace-nowrap">
                          <CopyableParty value={item.nameOrig} label="Sender" />
                          <span className="text-muted text-xs">&rarr;</span>
                          <CopyableParty value={item.nameDest} label="Recipient" />
                        </div>
                      </td>

                      {/* Amount with channel native currency symbol, right-aligned tabular-nums */}
                      <td className="py-2.5 px-3.5 text-right font-mono text-xs tabular-nums font-semibold text-text whitespace-nowrap">
                        {formatChannelAmount(item.amount, item.channel)}
                      </td>

                      {/* Score: number + thin bar, bar colour by band */}
                      <td className="py-2.5 px-3.5 whitespace-nowrap">
                        <div className="flex items-center gap-2 min-w-[110px]">
                          <div className="flex-1 h-1.5 rounded-full bg-border-strong/40 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${barColor} transition-[width] duration-300 ease-out`}
                              style={{ width: `${scorePct}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs tabular-nums text-text w-9 text-right font-medium">
                            {scoreClamped.toFixed(2)}
                          </span>
                        </div>
                      </td>

                      {/* Risk Band Badge */}
                      <td className="py-2.5 px-3.5 whitespace-nowrap">
                        <Badge variant={item.risk_band.toLowerCase() as 'low' | 'medium' | 'high'}>
                          {item.risk_band}
                        </Badge>
                      </td>

                      {/* Reason short text column from API */}
                      <td className="py-2.5 px-3.5 text-muted text-xs truncate max-w-[280px]">
                        {item.reason || 'Standard payment profile'}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Mobile Stacked Cards View (No horizontal scroll, key fields only, tap targets >= 44px) */}
        <div className="md:hidden p-3 space-y-2.5">
          {isLoading && items.length === 0 ? (
            Array.from({ length: 4 }).map((_, rIdx) => (
              <div key={`m-tx-skel-${rIdx}`} className="p-3.5 rounded-lg border border-border bg-surface space-y-2 animate-pulse">
                <div className="flex justify-between"><Skeleton className="h-4 w-24" /><Skeleton className="h-4 w-16" /></div>
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-full" />
              </div>
            ))
          ) : items.length === 0 ? (
            <div className="p-8 text-center text-xs text-muted">No transactions match your filters.</div>
          ) : (
            items.slice(0, 50).map((item, idx) => {
              const scoreClamped = Math.max(0, Math.min(1, item.risk_score));
              const scorePct = Math.round(scoreClamped * 100);
              const barColor =
                item.risk_band === 'High'
                  ? 'bg-danger'
                  : item.risk_band === 'Medium'
                  ? 'bg-warn'
                  : 'bg-accent';

              return (
                <div
                  key={`m-tx-${item.channel}-${item.step}-${idx}`}
                  onClick={() => setDetailTx(item)}
                  className="p-3.5 rounded-lg border border-border bg-surface hover:bg-surface-subtle active:bg-surface-subtle transition-colors cursor-pointer space-y-2 min-h-[44px]"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-text">{getChannelReadableName(item.channel)}</span>
                    <Badge variant={item.risk_band.toLowerCase() as 'low' | 'medium' | 'high'}>
                      {item.risk_band}
                    </Badge>
                  </div>

                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-mono text-sm font-semibold tabular-nums text-text">
                      {formatChannelAmount(item.amount, item.channel)}
                    </span>
                    <div className="flex items-center gap-1.5 font-mono text-xs text-muted">
                      <span>Score</span>
                      <span className="font-semibold text-text">{scoreClamped.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* Score bar */}
                  <div className="w-full h-1.5 rounded-full bg-border-strong/40 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${barColor}`}
                      style={{ width: `${scorePct}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-xs text-muted font-mono pt-1">
                    <span className="truncate max-w-[140px]">{item.nameOrig}</span>
                    <span>&rarr;</span>
                    <span className="truncate max-w-[140px] text-right">{item.nameDest}</span>
                  </div>

                  {item.reason && (
                    <div className="text-xs text-muted truncate pt-0.5">
                      {item.reason}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Minimal Sticky Pagination / Counts Footer */}
        <div className="p-3.5 border-t border-border flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-muted font-sans bg-surface">
          <div>
            Showing{' '}
            <span className="font-mono tabular-nums text-text font-semibold">
              {(page - 1) * pageSize + 1}
            </span>
            –
            <span className="font-mono tabular-nums text-text font-semibold">
              {Math.min(page * pageSize, data?.total || 0)}
            </span>{' '}
            of{' '}
            <span className="font-mono tabular-nums text-text font-semibold">
              {data?.total?.toLocaleString() ?? 0}
            </span>{' '}
            transactions
            <span className="ml-2 text-muted hidden md:inline">
              (Use <kbd className="font-mono px-1 py-0.5 rounded border border-border text-xs bg-surface-subtle">↑</kbd> <kbd className="font-mono px-1 py-0.5 rounded border border-border text-xs bg-surface-subtle">↓</kbd> to navigate, <kbd className="font-mono px-1 py-0.5 rounded border border-border text-xs bg-surface-subtle">Enter</kbd> to inspect)
            </span>
          </div>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              disabled={page <= 1}
              onClick={() => {
                setPage((p) => Math.max(1, p - 1));
                setSelectedRowIndex(-1);
              }}
            >
              Previous
            </Button>
            <span className="px-2 font-mono tabular-nums">
              Page {page} of {totalPages}
            </span>
            <Button
              variant="secondary"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => {
                setPage((p) => p + 1);
                setSelectedRowIndex(-1);
              }}
            >
              Next
            </Button>
          </div>
        </div>
      </div>

      {/* Right-Side Sheet Drawer for Row Factor Breakdown */}
      <Sheet open={!!detailTx} onOpenChange={(open) => !open && setDetailTx(null)}>
        <SheetContent side="right" className="overflow-y-auto">
          {detailTx && (
            <div className="space-y-6 text-xs">
              <SheetHeader>
                <div className="flex items-center gap-2 text-xs text-muted mb-1 font-mono">
                  <span>Step {detailTx.step}</span>
                  <span>&bull;</span>
                  <span>{getChannelReadableName(detailTx.channel)}</span>
                </div>
                <SheetTitle>Transaction Factor Breakdown</SheetTitle>
                <SheetDescription>
                  Detailed scoring parameters and TreeSHAP feature attributions for this payment.
                </SheetDescription>
              </SheetHeader>

              {/* Top Risk Pill & Score */}
              <div className="p-4 rounded-xl border border-border bg-surface-subtle flex items-center justify-between">
                <div>
                  <div className="text-xs text-muted">Calibrated Risk Score</div>
                  <div className="text-2xl font-semibold font-sans tracking-normal tabular-nums text-text mt-0.5">
                    {detailTx.risk_score.toFixed(4)}
                  </div>
                </div>
                <Badge variant={detailTx.risk_band.toLowerCase() as 'low' | 'medium' | 'high'}>
                  {detailTx.risk_band} Risk
                </Badge>
              </div>

              {/* Core Parameters */}
              <div className="space-y-3">
                <div className="font-semibold text-xs text-text">Parameters &amp; Counterparties</div>
                <div className="space-y-2 font-mono">
                  <div className="p-2.5 rounded-lg border border-border bg-surface-subtle flex justify-between items-center">
                    <span className="text-muted font-sans">Amount:</span>
                    <span className="font-semibold text-text tabular-nums">
                      {formatChannelAmount(detailTx.amount, detailTx.channel)}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg border border-border bg-surface-subtle flex justify-between items-center">
                    <span className="text-muted font-sans">Type:</span>
                    <span className="text-text">{detailTx.type}</span>
                  </div>
                  <div className="p-2.5 rounded-lg border border-border bg-surface-subtle flex justify-between items-center">
                    <span className="text-muted font-sans">Sender ID:</span>
                    <CopyableParty value={detailTx.nameOrig} label="Sender" />
                  </div>
                  <div className="p-2.5 rounded-lg border border-border bg-surface-subtle flex justify-between items-center">
                    <span className="text-muted font-sans">Recipient ID:</span>
                    <CopyableParty value={detailTx.nameDest} label="Recipient" />
                  </div>
                </div>
              </div>

              {/* Attributed Driver Reasons */}
              <div className="space-y-2">
                <div className="font-semibold text-xs text-text flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5 text-muted" />
                  <span>Attributed Risk Factors</span>
                </div>
                <div className="space-y-1.5">
                  {detailTx.top_reasons ? (
                    (Array.isArray(detailTx.top_reasons)
                      ? detailTx.top_reasons
                      : String(detailTx.top_reasons).split(';')
                    ).map((r: string, i: number) => (
                      <div
                        key={i}
                        className="p-2.5 rounded-lg border border-border bg-surface-subtle text-xs text-text leading-snug flex items-start gap-2"
                      >
                        <span className="w-1.5 h-1.5 rounded-full bg-accent mt-1.5 shrink-0" />
                        <span>{r.trim()}</span>
                      </div>
                    ))
                  ) : (
                    <div className="p-2.5 rounded-lg border border-border bg-surface-subtle text-xs text-muted">
                      {detailTx.reason || 'Nominal payment parameters; no abnormal risk drivers detected.'}
                    </div>
                  )}
                </div>
              </div>

              {/* Quick Navigation to Entity View */}
              <div className="pt-2">
                <Button
                  variant="primary"
                  size="sm"
                  className="w-full justify-center min-h-[38px]"
                  onClick={() => {
                    const targetAcc = detailTx.nameOrig || detailTx.nameDest || '';
                    setDetailTx(null);
                    navigate(`/entity?account=${encodeURIComponent(targetAcc)}`);
                  }}
                >
                  <Network className="w-3.5 h-3.5 mr-2" />
                  <span>Inspect Counterparty in Entity View</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
                </Button>
              </div>
            </div>
          )}
        </SheetContent>
      </Sheet>
    </div>
  );
};
