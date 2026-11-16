import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { visualizer } from 'rollup-plugin-visualizer';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'dist/stats.html',
      gzipSize: true,
      brotliSize: true,
      open: false,
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '^/api(/|$)': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/score': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/thresholds': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('three') || id.includes('@react-three')) {
              return 'vendor-three';
            }
            if (id.includes('echarts') || id.includes('zrender')) {
              return 'vendor-echarts';
            }
            if (id.includes('lucide-react')) {
              return 'vendor-lucide';
            }
            if (
              id.includes('@radix-ui') ||
              id.includes('cmdk') ||
              id.includes('vaul') ||
              id.includes('sonner')
            ) {
              return 'vendor-radix';
            }
            // Keep GSAP and Lenis strictly co-located within the landing chunk
            if (id.includes('gsap') || id.includes('lenis')) {
              return undefined;
            }
            if (id.includes('framer-motion')) {
              return 'vendor-motion';
            }
            if (
              id.includes('react') ||
              id.includes('react-dom') ||
              id.includes('react-router-dom') ||
              id.includes('@tanstack')
            ) {
              return 'vendor-react';
            }
          }
        },
      },
    },
  },
});

