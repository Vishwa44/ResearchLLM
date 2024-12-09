import React, { useState } from "react";
import { Menu, Search, Bell, User, MessageSquare, FileText } from "lucide-react";
import { Link } from "react-router-dom";

const NavBar = ({ toggleSidebar, onSearchClick, onNotificationsClick, onProfileClick }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    confirmPassword: "",
  });

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

          {/* Search Icon */}
          <button
            onClick={onSearchClick}
            className="hover:bg-zinc-800 rounded p-2"
          >
            <Search className="w-5 h-5 text-gray-300" />
          </button>

          {/* Notifications Icon */}
          <button
            onClick={onNotificationsClick}
            className="hover:bg-zinc-800 rounded p-2"
          >
            <Bell className="w-5 h-5 text-gray-300" />
          </button>

          {/* Profile Icon */}
          <button
            onClick={onProfileClick}
            className="hover:bg-zinc-800 rounded p-2"
          >
            <User className="w-5 h-5 text-gray-300" />
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
