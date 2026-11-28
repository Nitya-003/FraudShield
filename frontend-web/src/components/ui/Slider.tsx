import * as React from 'react';
import * as SliderPrimitive from '@radix-ui/react-slider';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
  return twMerge(clsx(inputs));
}

export interface SliderProps
  extends React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root> {
  label?: string;
  valueDisplay?: React.ReactNode;
}

export const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  SliderProps
>(({ className, label, valueDisplay, ...props }, ref) => (
  <div className="w-full space-y-2">
    {(label || valueDisplay) && (
      <div className="flex items-center justify-between text-xs">
        {label && <span className="font-semibold text-muted">{label}</span>}
        {valueDisplay && <span className="font-normal text-text tabular-nums">{valueDisplay}</span>}
      </div>
    )}
    <SliderPrimitive.Root
      ref={ref}
      className={cn(
        'relative flex w-full touch-none select-none items-center py-1.5 cursor-pointer',
        className
      )}
      {...props}
    >
      {/* Thicker 6px track for clear visibility */}
      <SliderPrimitive.Track className="relative h-2 w-full grow overflow-hidden rounded-full bg-border-strong/40 dark:bg-border-strong/60">
        <SliderPrimitive.Range className="absolute h-full bg-accent" />
      </SliderPrimitive.Track>
      {/* Prominent 18px thumb with clear focus ring */}
      <SliderPrimitive.Thumb
        aria-label={props['aria-label'] || (typeof label === 'string' ? label : 'Slider')}
        className="block h-4.5 w-4.5 rounded-full border-2 border-accent bg-surface shadow-card ring-offset-background transition-transform hover:scale-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
      />
    </SliderPrimitive.Root>
  </div>
));
Slider.displayName = SliderPrimitive.Root.displayName;
