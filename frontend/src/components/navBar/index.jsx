import React, { useState, useEffect } from "react";
import { Menu, Search, Bell, User, MessageSquare, FileText } from "lucide-react";
import { Link } from "react-router-dom";

const NavBar = ({ toggleSidebar, onSearchClick, onNotificationsClick, onProfileClick }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [selectedModel, setSelectedModel] = useState("llama3.2");

  useEffect(() => {
    const storedModel = localStorage.getItem("selectedModel") || "llama3.2";
    setSelectedModel(storedModel);
    localStorage.setItem("selectedModel", storedModel);
  }, []);

  const openModal = () => setIsModalOpen(true);
  const closeModal = () => setIsModalOpen(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleFormSubmit = (e) => {
    e.preventDefault();
    if (formData.password !== formData.confirmPassword) {
      alert("Passwords do not match!");
      return;
    }
    alert("Login/Register successful!");
    closeModal();
  };

  const handleModelChange = (e) => {
    const newModel = e.target.value;
    setSelectedModel(newModel);
    localStorage.setItem("selectedModel", newModel);
  };

  return (
    <>
      <div className="h-14 flex items-center px-4 border-b border-zinc-800 bg-zinc-900">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          {/* Sidebar Toggle */}
          {toggleSidebar && (
            <button
              onClick={toggleSidebar}
              className="hover:bg-zinc-800 rounded p-1"
            >
              <Menu className="w-6 h-6 text-gray-300" />
            </button>
          )}

          {/* Application Title */}
          <h1 className="text-lg font-semibold text-gray-300">
            <Link to="/search" className="hover:text-gray-200 text-gray-300">
              Research Paper Summarizer
            </Link>
          </h1>
        </div>

        {/* Right Section */}
        <div className="ml-auto flex items-center gap-4">
          {/* Dropdown */}
          <select
            value={selectedModel}
            onChange={handleModelChange}
            className="bg-zinc-900 text-gray-300 border border-zinc-700 rounded p-2"
          >
            <option value="llama3.2">llama3.2</option>
            <option value="llama3.3">llama3.3</option>
            <option value="gemini1.5">gemini1.5</option>
          </select>

          <Link to="/search" className="hover:text-gray-200 text-gray-300">
            <FileText className="inline-block w-5 h-5 mr-1" />
            Query
          </Link>
          <Link to="/notebook" className="hover:text-gray-200 text-gray-300">
            <MessageSquare className="inline-block w-5 h-5 mr-1" />
            Chat
          </Link>

          {/* Login Button */}
          <button
            onClick={openModal}
            className="hover:bg-zinc-800 rounded p-2 text-gray-300"
          >
            Login
          </button>
        </div>
      </div>

      {/* Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-zinc-800 text-gray-300 p-6 rounded-md shadow-lg w-96">
            <h2 className="text-lg font-bold mb-4">Login / Register</h2>
            <form onSubmit={handleFormSubmit} className="flex flex-col gap-4">
              <input
                type="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="Email"
                className="bg-zinc-900 text-gray-300 border border-zinc-700 rounded p-2"
                required
              />
              <input
                type="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Password"
                className="bg-zinc-900 text-gray-300 border border-zinc-700 rounded p-2"
                required
              />
              <input
                type="password"
                name="confirmPassword"
                value={formData.confirmPassword}
                onChange={handleInputChange}
                placeholder="Confirm Password"
                className="bg-zinc-900 text-gray-300 border border-zinc-700 rounded p-2"
                required
              />
              <button
                type="submit"
                className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded"
              >
                Submit
              </button>
            </form>
            <button
              onClick={closeModal}
              className="mt-4 text-gray-300 hover:underline"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default NavBar;
