import React from 'react';
import * as PopoverPrimitive from '@radix-ui/react-popover';

export interface PopoverProps {
  trigger: React.ReactNode;
  children: React.ReactNode;
  align?: 'start' | 'center' | 'end';
  side?: 'top' | 'right' | 'bottom' | 'left';
  className?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export const Popover: React.FC<PopoverProps> = ({
  trigger,
  children,
  align = 'center',
  side = 'bottom',
  className = '',
  open,
  onOpenChange,
}) => {
  return (
    <PopoverPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <PopoverPrimitive.Trigger asChild>
        {trigger}
      </PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          align={align}
          side={side}
          sideOffset={6}
          className={`z-50 w-72 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] p-4 text-xs text-[#0F172A] dark:text-[#E8EDF2] shadow-xl outline-none transition-[opacity,transform] duration-120 ${className}`}
        >
          {children}
          <PopoverPrimitive.Arrow className="fill-[#E3E8EE] dark:fill-[#1E2630]" />
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
};
