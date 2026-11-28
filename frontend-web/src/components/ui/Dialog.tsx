import React from 'react';
import * as DialogPrimitive from '@radix-ui/react-dialog';
import { X } from 'lucide-react';

export interface DialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description?: string;
  children: React.ReactNode;
}

export const Dialog: React.FC<DialogProps> = ({
  open,
  onOpenChange,
  title,
  description,
  children,
}) => {
  return (
    <DialogPrimitive.Root open={open} onOpenChange={onOpenChange}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm transition-opacity duration-200" />
        <DialogPrimitive.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-lg border border-[#E3E8EE] dark:border-[#1E2630] bg-white dark:bg-[#11171E] p-6 shadow-2xl outline-none text-[#0F172A] dark:text-[#E8EDF2] duration-200">
          <div className="flex items-start justify-between mb-4">
            <div>
              <DialogPrimitive.Title className="text-base font-semibold text-text tracking-normal font-sans">
                {title}
              </DialogPrimitive.Title>
              {description && (
                <DialogPrimitive.Description className="text-xs text-[#52606D] dark:text-[#8A96A3] mt-1 font-sans">
                  {description}
                </DialogPrimitive.Description>
              )}
            </div>
            <DialogPrimitive.Close className="rounded-md p-1.5 text-[#52606D] hover:text-[#0F172A] hover:bg-slate-100 dark:text-[#8A96A3] dark:hover:text-[#E8EDF2] dark:hover:bg-[#161F28] transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500/50">
              <X className="w-4 h-4" strokeWidth={1.5} />
            </DialogPrimitive.Close>
          </div>
          <div>{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
};
