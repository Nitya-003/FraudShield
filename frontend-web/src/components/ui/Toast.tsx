/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useContext, useCallback } from 'react';
import { Toaster as SonnerToaster, toast as sonnerToast } from 'sonner';

export type ToastVariant = 'info' | 'success' | 'warning' | 'error';

export interface ToastMessage {
  title: string;
  description?: string;
  variant?: ToastVariant;
  duration?: number;
}

interface ToastContextType {
  toast: (options: ToastMessage) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const toast = useCallback(({ title, description, variant = 'info', duration = 3500 }: ToastMessage) => {
    switch (variant) {
      case 'success':
        sonnerToast.success(title, { description, duration });
        break;
      case 'warning':
        sonnerToast.warning(title, { description, duration });
        break;
      case 'error':
        sonnerToast.error(title, { description, duration });
        break;
      case 'info':
      default:
        sonnerToast.info(title, { description, duration });
        break;
    }
  }, []);

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <SonnerToaster
        position="bottom-right"
        toastOptions={{
          className:
            'border font-sans text-xs shadow-overlay rounded-md duration-200 transition-[transform,opacity] border-border bg-surface text-text',
        }}
      />
    </ToastContext.Provider>
  );
};

export const useToast = (): ToastContextType => {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    return {
      toast: ({ title, description, variant }: ToastMessage) => {
        if (variant === 'success') sonnerToast.success(title, { description });
        else if (variant === 'error') sonnerToast.error(title, { description });
        else if (variant === 'warning') sonnerToast.warning(title, { description });
        else sonnerToast.info(title, { description });
      },
    };
  }
  return ctx;
};
