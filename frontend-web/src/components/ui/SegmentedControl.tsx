export interface SegmentOption<T extends string = string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string = string> {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
  size?: 'sm' | 'md';
}

export function SegmentedControl<T extends string = string>({
  options,
  value,
  onChange,
  className = '',
  size = 'md',
}: SegmentedControlProps<T>) {
  const sizeStyles = {
    sm: 'p-0.5 text-xs',
    md: 'p-1 text-xs',
  }[size];

  const itemSizeStyles = {
    sm: 'px-2 py-1',
    md: 'px-3 py-1.5',
  }[size];

  return (
    <div
      className={`inline-flex rounded-md border border-border bg-surface ${sizeStyles} ${className}`}
    >
      {options.map((opt) => {
        const isSelected = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`rounded font-medium transition-[color,background-color] duration-120 ${itemSizeStyles} ${
              isSelected
                ? 'bg-accent text-white dark:text-[#0B0F14] shadow-sm'
                : 'text-muted hover:text-text'
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
