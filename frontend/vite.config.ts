import { defineConfig } from "vite";

// Dev server proxies API calls to the FastAPI backend; production build is served by the backend itself.
export default defineConfig({
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8080", "/health": "http://127.0.0.1:8080", "/demo": "http://127.0.0.1:8080" } },
  build: { outDir: "dist", emptyOutDir: true, sourcemap: false },
});
