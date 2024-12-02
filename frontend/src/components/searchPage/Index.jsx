import React, { useState } from "react";

const DotPattern = () => (
  <svg className="absolute w-full h-full" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <pattern
        id="dot-pattern"
        x="0"
        y="0"
        width="20"
        height="20"
        patternUnits="userSpaceOnUse"
      >
        <circle
          cx="2"
          cy="2"
          r="1"
          fill="currentColor"
          className="text-zinc-800"
        />
      </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#dot-pattern)" />
  </svg>
);

const SearchingComponent = () => {
  const [searchText, setSearchText] = useState("");
  const [resultText, setResultText] = useState("Your summarized text will appear here...");

  const handleSearch = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: searchText }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch results. Please try again.");
      }

      const data = await response.json();
      setResultText(data.result || "No results found."); 
    } catch (error) {
      setResultText(`Error: ${error.message}`);
    }
  };


  return (
    <div className="flex h-screen bg-zinc-900 text-gray-300">
      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Navbar */}
        <div className="h-14 flex items-center px-4 border-b border-zinc-800 bg-zinc-800">
          <h1 className="text-lg font-semibold">Research Paper Summarizer</h1>
        </div>

        {/* Content Area */}
        <div className="flex-1 relative p-6">
          {/* Dot Pattern */}
          <DotPattern />

          {/* Search and Results Section */}
          <div className="relative z-10 flex flex-col items-center gap-6 mt-8 mx-auto w-full max-w-3xl">
            {/* Search Section */}
            <form
              onSubmit={handleSearch}
              className="flex items-center bg-zinc-800 p-4 rounded-lg shadow-md w-full"
            >
              <input
                type="text"
                placeholder="Enter your query here..."
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                className="flex-1 px-4 py-2 text-sm text-gray-300 bg-transparent focus:outline-none"
              />
              <button
                type="submit"
                className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-sm rounded-md"
              >
                Search
              </button>
            </form>

            {/* Results Block */}
            <div className="bg-zinc-800 p-6 rounded-lg shadow-md w-full">
              <h2 className="text-lg font-semibold mb-4">Results</h2>
              <p className="text-sm text-gray-400">{resultText}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SearchingComponent;
