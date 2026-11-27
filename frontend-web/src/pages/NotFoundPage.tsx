import React from 'react';
import { Link } from 'react-router-dom';
import { FileQuestion, Home, ArrowLeft } from 'lucide-react';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
      <div className="w-16 h-16 rounded-full bg-surface-subtle text-muted flex items-center justify-center mb-6">
        <FileQuestion className="w-8 h-8" />
      </div>
      <span className="font-mono text-xs text-muted mb-2">
        Error 404
      </span>
      <h1 className="font-sans font-semibold text-xl text-text tracking-normal mb-3">
        Page Not Found
      </h1>
      <p className="text-sm text-muted max-w-md mb-8">
        The requested route does not exist. It may have been moved, renamed, or was an invalid deep link.
      </p>
      <div className="flex items-center gap-3">
        <button
          onClick={() => window.history.back()}
          className="inline-flex items-center gap-2 px-4 py-2 rounded text-xs font-semibold border border-border text-text hover:bg-surface-subtle transition-colors"
        >
          <ArrowLeft className="w-3.5 h-3.5" />
          <span>Go Back</span>
        </button>
        <Link
          to="/"
          className="inline-flex items-center gap-2 px-4 py-2 rounded text-xs font-semibold bg-accent text-white dark:text-[#0B0F14] hover:opacity-90 transition-opacity"
        >
          <Home className="w-3.5 h-3.5" />
          <span>Return to Dashboard</span>
        </Link>
      </div>
    </div>
  );
};

