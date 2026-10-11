import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { DEFAULT: "#0f1b2d", 2: "#52606d", 3: "#8a96a3" },
        brand: { 50: "#ecfdf6", 100: "#d1fae8", 500: "#10a37f", 600: "#0b8a6b", 700: "#08664f", 900: "#0a2a22" },
        navy: { 800: "#13233a", 900: "#0c1828" },
        surface: "#f6f7f9",
        series: { 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100" },
      },
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
export default config;
