import React from 'react';
import { Drawer as VaulDrawer } from 'vaul';
import { X } from 'lucide-react';

export interface DrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title?: string;
  description?: string;
  children: React.ReactNode;
  direction?: 'right' | 'bottom';
}

export const Drawer: React.FC<DrawerProps> = ({
  open,
  onOpenChange,
  title,
  description,
  children,
  direction = 'right',
}) => {
  return (
    <VaulDrawer.Root
      open={open}
      onOpenChange={onOpenChange}
      direction={direction}
      dismissible={true}
    >
      <VaulDrawer.Portal>
        <VaulDrawer.Overlay className="fixed inset-0 z-50 bg-black/40 backdrop-blur-[2px] transition-opacity duration-200 ease-control" />
        <VaulDrawer.Content
          className={`fixed z-50 bg-surface border-border text-text shadow-overlay transition-transform duration-200 ease-control outline-none ${
            direction === 'right'
              ? 'top-0 right-0 bottom-0 w-full max-w-md border-l h-full flex flex-col'
              : 'bottom-0 left-0 right-0 max-h-[85vh] rounded-t-lg border-t flex flex-col'
          }`}
        >
          {/* Header */}
          <div className="p-5 border-b border-border flex items-start justify-between shrink-0">
            <div>
              {title && (
                <VaulDrawer.Title className="text-base font-semibold text-text tracking-normal font-sans">
                  {title}
                </VaulDrawer.Title>
              )}
              {description && (
                <VaulDrawer.Description className="text-xs text-muted mt-1 font-sans font-normal">
                  {description}
                </VaulDrawer.Description>
              )}
            </div>
            <VaulDrawer.Close className="rounded-sm p-1.5 text-muted hover:text-text hover:bg-surface-subtle transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent">
              <X className="w-4 h-4" strokeWidth={1.5} />
            </VaulDrawer.Close>
          </div>

          {/* Body Content */}
          <div className="flex-1 overflow-y-auto p-5 text-sm text-text">
            {children}
          </div>
        </VaulDrawer.Content>
      </VaulDrawer.Portal>
    </VaulDrawer.Root>
  );
};

