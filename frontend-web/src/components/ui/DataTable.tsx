import React, { useState } from 'react';
import {
  ColumnDef,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  RowSelectionState,
} from '@tanstack/react-table';
import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

interface DataTableProps<TData> {
  columns: ColumnDef<TData, unknown>[];
  data: TData[];
  enableRowSelection?: boolean;
  onSelectionChange?: (selectedRows: TData[]) => void;
  onRowClick?: (row: TData) => void;
  isLoading?: boolean;
  emptyMessage?: string;
  className?: string;
}

export function DataTable<TData>({
  columns,
  data,
  enableRowSelection = false,
  onSelectionChange,
  onRowClick,
  isLoading = false,
  emptyMessage = 'No records found',
  className = '',
}: DataTableProps<TData>) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [rowSelection, setRowSelection] = useState<RowSelectionState>({});

  const table = useReactTable({
    data,
    columns,
    state: {
      sorting,
      rowSelection,
    },
    enableRowSelection,
    onSortingChange: setSorting,
    onRowSelectionChange: (updaterOrValue) => {
      setRowSelection((old) => {
        const next = typeof updaterOrValue === 'function' ? updaterOrValue(old) : updaterOrValue;
        if (onSelectionChange) {
          const selected = Object.keys(next).map((idx) => data[Number(idx)]).filter(Boolean);
          onSelectionChange(selected);
        }
        return next;
      });
    },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className={`overflow-x-auto border border-border rounded-lg bg-surface shadow-card ${className}`}>
      <table className="w-full min-w-[720px] text-left text-xs border-collapse">
        {/* Sticky Header */}
        <thead className="bg-surface-subtle border-b border-border sticky top-0 z-10">
          {table.getHeaderGroups().map((headerGroup) => (
            <tr key={headerGroup.id}>
              {headerGroup.headers.map((header, cIdx) => {
                const canSort = header.column.getCanSort();
                const isSorted = header.column.getIsSorted();
                return (
                  <th
                    key={header.id}
                    scope="col"
                    className={`py-2.5 px-3 font-semibold tracking-normal text-xs text-muted select-none ${
                      cIdx === 0
                        ? 'sticky left-0 bg-surface-subtle z-20 shadow-[1px_0_0_0_var(--border)]'
                        : ''
                    }`}
                  >
                    {header.isPlaceholder ? null : (
                      <div
                        className={`flex items-center gap-1.5 ${
                          canSort ? 'cursor-pointer hover:text-text' : ''
                        }`}
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        <span className="capitalize">{flexRender(header.column.columnDef.header, header.getContext())}</span>
                        {canSort && (
                          <span className="shrink-0 text-muted">
                            {isSorted === 'asc' ? (
                              <ArrowUp className="w-3.5 h-3.5 text-text" />
                            ) : isSorted === 'desc' ? (
                              <ArrowDown className="w-3.5 h-3.5 text-text" />
                            ) : (
                              <ArrowUpDown className="w-3.5 h-3.5 opacity-40 hover:opacity-100" />
                            )}
                          </span>
                        )}
                      </div>
                    )}
                  </th>
                );
              })}
            </tr>
          ))}
        </thead>

        <tbody className="divide-y divide-border font-sans">
          {isLoading && data.length === 0 ? (
            Array.from({ length: 6 }).map((_, rIdx) => (
              <tr key={`skeleton-${rIdx}`} className="animate-pulse">
                {columns.map((_, cIdx) => (
                  <td
                    key={`skel-cell-${cIdx}`}
                    className={`py-3 px-3 ${
                      cIdx === 0
                        ? 'sticky left-0 bg-white dark:bg-[#11171E] z-10 shadow-[1px_0_0_0_#E3E8EE] dark:shadow-[1px_0_0_0_#1E2630]'
                        : ''
                    }`}
                  >
                    <div
                      className="h-3.5 bg-slate-200 dark:bg-[#1E2630] rounded"
                      style={{ width: `${40 + ((rIdx * 17 + cIdx * 23) % 45)}%` }}
                    />
                  </td>
                ))}
              </tr>
            ))
          ) : table.getRowModel().rows.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="py-12 text-center text-[#52606D] dark:text-[#8A96A3]">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                onClick={() => onRowClick?.(row.original)}
                className={`group transition-colors duration-120 hover:bg-slate-50 dark:hover:bg-[#161F28] ${
                  onRowClick ? 'cursor-pointer' : ''
                } ${
                  row.getIsSelected() ? 'bg-slate-100 dark:bg-[#1E2630]' : ''
                }`}
              >
                {row.getVisibleCells().map((cell, cIdx) => (
                  <td
                    key={cell.id}
                    className={`py-2.5 px-3 text-[#0F172A] dark:text-[#E8EDF2] whitespace-nowrap ${
                      cIdx === 0
                        ? 'sticky left-0 bg-white dark:bg-[#11171E] group-hover:bg-slate-50 dark:group-hover:bg-[#161F28] z-10 shadow-[1px_0_0_0_#E3E8EE] dark:shadow-[1px_0_0_0_#1E2630]'
                        : ''
                    }`}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Reusable Inline Risk Progress Bar for Table Cells
 */
export const InlineRiskBar: React.FC<{ score: number; showScore?: boolean }> = ({
  score,
  showScore = true,
}) => {
  const clamped = Math.max(0, Math.min(1, score));
  const percent = Math.round(clamped * 100);

  const colorClass =
    clamped >= 0.85
      ? 'bg-rose-500'
      : clamped >= 0.45
      ? 'bg-amber-500'
      : 'bg-emerald-500';

  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 h-1.5 rounded-full bg-slate-100 dark:bg-[#161F28] overflow-hidden">
        <div
          className={`h-full rounded-full ${colorClass} transition-[width] duration-300 ease-out motion-reduce:transition-none`}
          style={{ width: `${percent}%` }}
        />
      </div>
      {showScore && (
        <span className="font-mono text-xs tabular-nums text-[#0F172A] dark:text-[#E8EDF2] w-9 text-right font-medium">
          {clamped.toFixed(2)}
        </span>
      )}
    </div>
  );
};
