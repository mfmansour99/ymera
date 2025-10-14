import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { visualizer } from 'rollup-plugin-visualizer';
import { resolve } from 'path';

export default defineConfig(({ mode }) => {
  // Load env file based on mode
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
    plugins: [
      react({
        // Enable Fast Refresh
        fastRefresh: true,
        // Babel configuration for optimizations
        babel: {
          plugins: [
            // Add any babel plugins here if needed
          ],
        },
      }),
      // Bundle analyzer (only in analyze mode)
      mode === 'analyze' && visualizer({
        open: true,
        gzipSize: true,
        brotliSize: true,
        filename: './dist/stats.html',
      }),
    ].filter(Boolean),

    base: '/',

    build: {
      outDir: 'dist',
      emptyOutDir: true,
      // Enable/disable sourcemaps based on environment
      sourcemap: env.ENABLE_SOURCEMAPS === 'true' ? true : false,
      minify: 'terser',
      terserOptions: {
        compress: {
          // Remove console.log in production
          drop_console: mode === 'production',
          drop_debugger: true,
          passes: 2,
          pure_funcs: mode === 'production' ? ['console.log', 'console.info', 'console.debug'] : [],
        },
        mangle: {
          safari10: true, // Fix Safari 10/11 bugs
        },
        format: {
          comments: false, // Remove all comments
        },
      },
      rollupOptions: {
        output: {
          // Manual chunks for better caching
          manualChunks: {
            // Vendor chunks
            'react-vendor': ['react', 'react-dom'],
            'three-vendor': ['three', '@react-three/fiber', '@react-three/drei'],
            'animation-vendor': ['gsap'],
            
            // Utility chunks
            'utils': [
              'lodash', 
              'date-fns',
            ],
            
            // Icon libraries
            'icons': [
              'lucide-react',
            ],
          },
          // Asset naming
          chunkFileNames: 'assets/js/[name]-[hash].js',
          entryFileNames: 'assets/js/[name]-[hash].js',
          assetFileNames: 'assets/[ext]/[name]-[hash].[ext]',
        },
        // External dependencies (if any)
        external: [],
      },
      // Chunk size warning limit (500kb)
      chunkSizeWarningLimit: 500,
      // Report compressed size
      reportCompressedSize: true,
      // CSS code splitting
      cssCodeSplit: true,
      // Asset inline limit (4kb)
      assetsInlineLimit: 4096,
    },

    server: {
      port: parseInt(env.VITE_PORT || '3000'),
      open: env.VITE_OPEN_BROWSER !== 'false',
      strictPort: false, // Allow fallback to next available port
      host: env.VITE_HOST || 'localhost',
      cors: true,
      hmr: {
        overlay: true, // Show error overlay
        protocol: 'ws',
      },
      // Proxy configuration for API calls (if needed)
      proxy: env.VITE_API_PROXY ? {
        '/api': {
          target: env.VITE_API_PROXY,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      } : undefined,
    },

    preview: {
      port: parseInt(env.VITE_PREVIEW_PORT || '4173'),
      open: true,
      strictPort: false,
      host: env.VITE_HOST || 'localhost',
    },

    resolve: {
      alias: {
        '@': resolve(__dirname, './src'),
        '@components': resolve(__dirname, './src/components'),
        '@utils': resolve(__dirname, './src/utils'),
        '@types': resolve(__dirname, './src/types'),
        '@hooks': resolve(__dirname, './src/hooks'),
        '@assets': resolve(__dirname, './src/assets'),
        '@styles': resolve(__dirname, './src/styles'),
      },
      extensions: ['.mjs', '.js', '.ts', '.jsx', '.tsx', '.json'],
    },

    css: {
      preprocessorOptions: {
        scss: {
          additionalData: `@import "@/styles/variables.scss";`,
        },
      },
      postcss: {
        plugins: [
          require('autoprefixer')({
            overrideBrowserslist: [
              '> 1%',
              'last 2 versions',
              'not dead',
              'not ie 11',
            ],
          }),
          require('tailwindcss'),
          mode === 'production' && require('cssnano')({
            preset: ['default', {
              discardComments: {
                removeAll: true,
              },
            }],
          }),
        ].filter(Boolean),
      },
      modules: {
        localsConvention: 'camelCase',
        generateScopedName: mode === 'production' 
          ? '[hash:base64:5]'
          : '[name]__[local]__[hash:base64:5]',
      },
    },

    optimizeDeps: {
      include: [
        'react',
        'react-dom',
        'gsap',
        'three',
        '@react-three/fiber',
        '@react-three/drei',
        'date-fns',
        'lucide-react',
      ],
      exclude: [
        // Exclude any packages that should not be pre-bundled
      ],
    },

    // Define global constants
    define: {
      __APP_VERSION__: JSON.stringify(env.npm_package_version || '1.0.0'),
      __BUILD_TIME__: JSON.stringify(new Date().toISOString()),
      __DEV__: mode === 'development',
      __PROD__: mode === 'production',
    },

    // Environment variables
    envPrefix: 'VITE_',
    envDir: '.',

    // Enable JSON imports
    json: {
      namedExports: true,
      stringify: false,
    },

    // Esbuild options for faster builds
    esbuild: {
      logOverride: { 'this-is-undefined-in-esm': 'silent' },
      legalComments: 'none',
      // Target modern browsers for smaller bundle
      target: 'es2020',
      // Drop console/debugger in production
      drop: mode === 'production' ? ['console', 'debugger'] : [],
    },

    // Performance optimizations
    performance: {
      // Disable performance hints for large chunks (handled by chunkSizeWarningLimit)
      hints: false,
    },
  };
});