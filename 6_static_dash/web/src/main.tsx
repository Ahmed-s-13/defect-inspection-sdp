import React from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/b612/400.css";
import "@fontsource/b612/700.css";
import "@fontsource/atkinson-hyperlegible-next/400.css";
import "@fontsource/atkinson-hyperlegible-next/500.css";
import "@fontsource/atkinson-hyperlegible-next/700.css";
import "./styles/tokens.css";
import "./styles/shell.css";
import { App } from "./App";

history.scrollRestoration = "manual"; // the app decides where each view starts (Batch keeps its place itself)

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
