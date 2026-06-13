import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    // Model download + inference can be slow on a cold cache.
    testTimeout: 600_000,
    hookTimeout: 600_000,
  },
});
