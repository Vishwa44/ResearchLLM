import React from "react";
import { BrowserRouter, Routes, Route } from "react-router";

import NotebookPage from "./components/HomePage";

import "./index.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<NotebookPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
