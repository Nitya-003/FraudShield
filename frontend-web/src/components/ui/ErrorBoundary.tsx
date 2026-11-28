import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
  onReset?: () => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error inside FraudShield component tree:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="w-full p-8 rounded-lg border border-red-200 dark:border-red-900/40 bg-red-50/50 dark:bg-red-950/20 text-center my-6">
          <div className="w-12 h-12 rounded-full bg-red-100 dark:bg-red-900/50 text-red-600 dark:text-red-400 mx-auto flex items-center justify-center mb-4">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <h3 className="font-sans font-semibold text-lg text-text mb-2">
            {this.props.fallbackTitle || 'Component Render Error'}
          </h3>
          <p className="text-sm text-muted max-w-md mx-auto mb-6">
            An unexpected error occurred while rendering this ledger view. Operational metrics and session data remain safe.
          </p>
          {this.state.error && (
            <div className="mb-6 p-3 rounded bg-surface border border-border font-mono text-xs text-danger max-w-xl mx-auto text-left overflow-x-auto">
              {this.state.error.toString()}
            </div>
          )}
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={this.handleReset}
              className="inline-flex items-center gap-2 px-4 py-2 rounded text-xs font-semibold bg-accent text-white dark:text-[#0B0F14] hover:opacity-90 transition-opacity"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Component</span>
            </button>
            <a
              href="/"
              className="inline-flex items-center gap-2 px-4 py-2 rounded text-xs font-semibold border border-border text-text hover:bg-surface-subtle transition-colors"
            >
              <Home className="w-3.5 h-3.5" />
              <span>Return Home</span>
            </a>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
