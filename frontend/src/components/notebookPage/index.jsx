import React, { useEffect, useState } from "react";
import { Menu, FileText } from "lucide-react";
import NavBar from "../navbar";

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
  const [sources, setSources] = useState([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedSourceId, setSelectedSourceId] = useState(null);
  const [searchText, setSearchText] = useState("");
  const [searchResult, setResultText] = useState("");

  // Fetch papers from the backend
  const fetchPapers = async () => {
    const userUUID = "user-123"; // Replace this with the actual user UUID
    try {
      const response = await fetch("http://127.0.0.1:5000/getPapers", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ uuid: userUUID }), // Send the UUID in the request body
      });
  
      if (!response.ok) {
        throw new Error("Failed to fetch papers");
      }
  
      const data = await response.json();
      setSources(data); // Update sources with the fetched data
    } catch (error) {
      console.error("Error fetching papers:", error.message);
    }
  };
  

  useEffect(() => {
    fetchPapers(); // Fetch papers when the component mounts
  }, []);

  const toggleSidebar = () => {
    setIsSidebarOpen(!isSidebarOpen);
  };

  const selectSource = (sourceId) => {
    setSelectedSourceId(sourceId === selectedSourceId ? null : sourceId);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      const formData = new FormData();
      formData.append("file", file);

      // Replace this with your file upload API endpoint
      const response = await fetch("http://127.0.0.1:5000/summarize", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("File upload failed");
      }

      const data = await response.json();
      // Add the uploaded file to the sources list
      const newSource = {
        name: file.name,
        type: file.type,
        id: sources.length + 1,
      };
      setSources([...sources, newSource]);
      setSearchText("");
      setResultText(data.message || "No summary available.");
      alert("File uploaded successfully!");
    } catch (error) {
      console.error("Upload error:", error);
      alert(`Error: ${error.message}`);
    }
  };


  const handleSearch = async (e) => {
    e.preventDefault();

    const selectedSource = sources.find((source) => source.id === selectedSourceId);
    if (!selectedSource) {
      alert("Please select a source to summarize.");
      return;
    }

    const selectedModel = localStorage.getItem("selectedModel") || "llama3.2";

    try {
      const payload = {
        file_id: selectedSource.id, // Use the file ID
        query: searchText,
        model: selectedModel, // Use the selected model
      };

      const response = await fetch("http://127.0.0.1:5000/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch results. Please try again.");
      }

      const data = await response.json();
      alert("Query fetched successfully!");
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
                <label htmlFor="file-upload" className="text-lg">+</label>
                <input
                  id="file-upload"
                  type="file"
                  onChange={handleFileUpload}
                  className="hidden"
                />

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
                    ${selectedSourceId === source.id
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
          <div className="flex flex-col items-center pt-4 gap-4">
            <button className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-zinc-800 border border-zinc-700">
              <span className="text-lg">+</span>
            </button>
            <div className="w-6 h-6 flex items-center justify-center rounded-full border border-zinc-700 text-sm">
              {sources.length}
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        <NavBar />
        <div className="flex-1 relative flex items-center justify-center">
          <DotPattern />
          <div className="relative text-zinc-500">
            {sources.length === 0
              ? "Add source to search"
              : searchResult.length > 0
              ? <div>{searchResult}</div>
              : "Add prompt to search through the selected source"}
          </div>
        </div>
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
              <button
                onClick={handleSearch}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-md bg-blue-500 hover:bg-blue-600"
              >
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
