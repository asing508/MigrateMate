"use client";

import { useState } from "react";
import { Github, ArrowRight, Zap, FileCode, CheckCircle } from "lucide-react";
import MigrationPanel from "@/components/MigrationPanel";

export default function Home() {
  const [mode, setMode] = useState<"landing" | "migrate">("landing");

  if (mode === "migrate") {
    return <MigrationPanel onBack={() => setMode("landing")} />;
  }

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Header */}
      <header className="border-b border-[var(--border)] px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold">MigrateMate</span>
          </div>

          <a
            href="https://github.com"
            target="_blank"
            className="text-gray-400 hover:text-white transition"
          >
            <Github className="w-6 h-6" />
          </a>
        </div>
      </header>

      {/* Hero Section */}
      <main className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Flask → FastAPI
          </h1>
          <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
            Automatically migrate your Flask applications to FastAPI using AI-powered code transformation.
            Supports entire GitHub repositories.
          </p>
          <button
            onClick={() => setMode("migrate")}
            className="inline-flex items-center gap-2 px-8 py-4 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white font-semibold rounded-lg transition text-lg"
          >
            Start Migration
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-6 mb-20">
          <FeatureCard
            icon={<Github className="w-8 h-8" />}
            title="GitHub Integration"
            description="Clone and migrate entire repositories with a single URL. Supports public repos."
          />
          <FeatureCard
            icon={<FileCode className="w-8 h-8" />}
            title="Smart Parsing"
            description="AST-based code analysis extracts functions, classes, and routes accurately."
          />
          <FeatureCard
            icon={<Zap className="w-8 h-8" />}
            title="AI-Powered"
            description="Uses Gemini AI with Plan→Code→Test→Fix workflow for reliable migrations."
          />
        </div>

        {/* How It Works */}
        <div className="mb-20">
          <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
          <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-8">
            <Step number={1} title="Input" description="Paste GitHub URL or upload ZIP" />
            <ArrowRight className="w-6 h-6 text-gray-600 hidden md:block" />
            <Step number={2} title="Analyze" description="Parse & extract code chunks" />
            <ArrowRight className="w-6 h-6 text-gray-600 hidden md:block" />
            <Step number={3} title="Migrate" description="AI converts each function" />
            <ArrowRight className="w-6 h-6 text-gray-600 hidden md:block" />
            <Step number={4} title="Download" description="Get your FastAPI project" />
          </div>
        </div>

        {/* Supported Conversions */}
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-6">Automatic Conversions</h2>
          <div className="flex flex-wrap justify-center gap-3">
            {[
              "@app.route → @app.get/post",
              "<int:id> → {id: int}",
              "jsonify() → direct return",
              "request.get_json() → Pydantic",
              "def → async def",
              "404 returns → HTTPException",
            ].map((item) => (
              <span
                key={item}
                className="inline-flex items-center gap-2 px-4 py-2 bg-[var(--card)] rounded-lg text-sm"
              >
                <CheckCircle className="w-4 h-4 text-green-500" />
                {item}
              </span>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--border)] px-6 py-8 mt-20">
        <div className="max-w-7xl mx-auto text-center text-gray-500 text-sm">
          Built with FastAPI, LangGraph, and Gemini AI
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
}) {
  return (
    <div className="p-6 bg-[var(--card)] rounded-xl border border-[var(--border)] hover:border-gray-600 transition">
      <div className="text-blue-500 mb-4">{icon}</div>
      <h3 className="text-lg font-semibold mb-2">{title}</h3>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}

function Step({
  number,
  title,
  description,
}: {
  number: number;
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col items-center text-center">
      <div className="w-12 h-12 rounded-full bg-[var(--primary)] flex items-center justify-center text-white font-bold text-lg mb-3">
        {number}
      </div>
      <h4 className="font-semibold mb-1">{title}</h4>
      <p className="text-gray-400 text-sm">{description}</p>
    </div>
  );
}