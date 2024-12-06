import React, { useState } from "react";
import { Menu, FileText } from "lucide-react";

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

const NotebookInterface = () => {
  const [sources, setSources] = useState([
    { name: "ResearchPaper.pdf", type: "pdf", id: 1 },
    { name: "ResearchPaper2.pdf", type: "pdf", id: 2 },
  ]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedSourceId, setSelectedSourceId] = useState(null);
  const [searchText, setSearchText] = useState("");
  const [searchResult, setResultText] = useState("")

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const selectSource = (sourceId) => {
    setSelectedSourceId(sourceId === selectedSourceId ? null : sourceId);
  };


  const handleSearch = async (e) => {
    e.preventDefault();
  
    // Find the selected source file
    const selectedSource = sources.find((source) => source.id === selectedSourceId);
  
    if (!selectedSource) {
      alert("Please select a source to summarize.");
      return;
    }
  
    try {
      // Create a FormData object to send the file
      const formData = new FormData();
      // This filePath needs to be updated with the selected files, file path can be retrieved from S3 in the backend
      const filePath = `https://s3.us-west-2.amazonaws.com/chatbotcloud.com/An+Optimal+Control+View+of+Adversarial+Machine+Learning.pdf`; // Adjust file path as needed
  
      // Append the file to the FormData object
      const fileBlob = await fetch(filePath).then((res) => res.blob());
      formData.append("file", fileBlob, selectedSource.name);
  
      // Make the POST request to the API
      const response = await fetch("http://127.0.0.1:5000/summarize", {
        method: "POST",
        body: formData,
      });
  
      if (!response.ok) {
        throw new Error("Failed to fetch results. Please try again.");
      }
  
      const data = await response.json();
      alert("Summary fetched successfully!");
      setSearchText("");
      setResultText(data.summary || "No summary available.");
    } catch (error) {
      alert(`Error: ${error.message}`);
    }
  };
  

  return (
    <div className="flex h-screen bg-zinc-900 text-gray-300">
      {/* Sidebar */}
      <div
        className={`
          ${isSidebarOpen ? "w-64" : "w-16"} 
          flex-shrink-0 transition-all duration-300 ease-in-out 
          border-r border-zinc-800 flex flex-col bg-zinc-900
        `}
      >
        {/* Header */}
        <div className="h-14 flex items-center px-4 border-b border-zinc-800">
          <div className="flex items-center gap-2">
            <button
              onClick={toggleSidebar}
              className="hover:bg-zinc-800 rounded p-1"
            >
              <Menu className="w-6 h-6" />
            </button>
          </div>
        </div>

        {isSidebarOpen ? (
          // Expanded Sidebar Content
          <div
            className="p-4 border-b border-zinc-800"
            style={{ borderBottom: "none" }}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="w-6 h-6 flex items-center justify-center rounded-full border border-zinc-700">
                  {sources.length}
                </div>
                <span>Sources</span>
              </div>
              <button className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-zinc-800 border border-zinc-700">
                <span className="text-lg">+</span>
              </button>
            </div>

            {sources.map((source) => (
              <div
                key={source.id}
                className="flex items-center justify-between py-2 px-3 text-sm hover:bg-zinc-800/50 rounded cursor-pointer"
                onClick={() => selectSource(source.id)}
                role="radio"
                aria-checked={selectedSourceId === source.id}
              >
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-red-500" />
                  <span>{source.name}</span>
                </div>
                <div className="relative flex items-center">
                  <div
                    className={`
                    w-4 h-4 rounded-full border
                    ${
                      selectedSourceId === source.id
                        ? "border-blue-500 bg-blue-500"
                        : "border-zinc-600"
                    }
                    transition-colors duration-200
                  `}
                  >
                    {selectedSourceId === source.id && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-1.5 h-1.5 bg-white rounded-full"></div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          // Collapsed Sidebar Content
          <div className="flex flex-col items-center pt-4 gap-4">
            {/* Add Source Button */}
            <button className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-zinc-800 border border-zinc-700">
              <span className="text-lg">+</span>
            </button>

            {/* Source Count */}
            <div className="w-6 h-6 flex items-center justify-center rounded-full border border-zinc-700 text-sm">
              {sources.length}
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <div className="h-14 flex items-center px-4 border-b border-zinc-800">
          <h1 className="text-lg">Research Paper Summarizer</h1>
        </div>

        {/* Empty State with Dot Pattern */}
        <div className="flex-1 relative flex items-center justify-center">
          <DotPattern />
          <div className="relative text-zinc-500">
            {sources.length == 0
              ? "Add source to search"
              : searchResult.length > 0 
              ? <div>{searchResult}</div> 
              : "Add prompt to search through the selected source"}

          </div>
        </div>

        {/* Bottom Chat Interface */}
        <div className="p-4 border-t border-zinc-800">
          <div className="max-w-4xl mx-auto">
            <div className="relative">
              <input
                type="text"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                placeholder="Start typing..."
                className="w-full bg-zinc-800 rounded-lg pl-4 pr-12 py-3 focus:outline-none focus:ring-1 focus:ring-blue-500"
              />
              <button onClick={handleSearch} className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-md bg-blue-500 hover:bg-blue-600">
                <svg
                  className="w-4 h-4 rotate-90"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default NotebookInterface;
