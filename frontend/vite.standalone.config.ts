import { defineConfig } from "vite";

// Standalone offline scene bundle: one IIFE script + one stylesheet that the backend inlines into a single HTML file
// (GET /api/jobs/{id}/export/scene.html). Built after the app into dist/standalone.
export default defineConfig({
  build: {
    outDir: "dist/standalone",
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 900,
    cssCodeSplit: false,
    lib: { entry: "src/standalone.ts", name: "DepthWizardScene", formats: ["iife"], fileName: () => "standalone.js", cssFileName: "standalone" },
  },
});
