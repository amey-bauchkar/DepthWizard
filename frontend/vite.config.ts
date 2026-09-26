import { defineConfig } from "vite";

// Dev server proxies API calls to the FastAPI backend; production build is served by the backend itself.
export default defineConfig({
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8000", "/health": "http://127.0.0.1:8000", "/demo": "http://127.0.0.1:8000" } },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false, chunkSizeWarningLimit: 900 }, // three.js is ~600 kB; served locally
});
