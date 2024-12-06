import React from "react";
import { BrowserRouter, Routes, Route } from "react-router";

import NotebookPage from "./components/notebookPage";
import SearchingComponent from "./components/searchPage/Index";

import "./index.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<SearchingComponent />} />
        <Route path="/notebook" element={<NotebookPage />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
