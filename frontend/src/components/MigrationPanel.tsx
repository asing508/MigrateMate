"use client";

import { useState, useEffect } from "react";
import {
    ArrowLeft,
    Github,
    Upload,
    Loader2,
    CheckCircle,
    XCircle,
    Download,
    FileCode,
    AlertCircle,
} from "lucide-react";
import axios from "axios";
import CodeDiffViewer from "./CodeDiffViewer";

const API_URL = "http://localhost:8000/api/v1";

interface MigrationResult {
    source_path: string;
    output_path: string;
    source_content: string;
    migrated_content: string;
    chunks_migrated: number;
    chunks_failed: number;
    confidence: number;
}

interface MigrationSummary {
    total_files: number;
    files_migrated: number;
    total_chunks?: number;
    chunks_succeeded?: number;
    chunks_failed?: number;
    average_confidence?: number;
}

type MigrationStatus =
    | "idle"
    | "cloning"
    | "analyzing"
    | "migrating"
    | "packaging"
    | "completed"
    | "failed";

export default function MigrationPanel({ onBack }: { onBack: () => void }) {
    const [inputType, setInputType] = useState<"github" | "upload">("github");
    const [repoUrl, setRepoUrl] = useState("");
    const [branch, setBranch] = useState("main");
    const [status, setStatus] = useState<MigrationStatus>("idle");
    const [progress, setProgress] = useState(0);
    const [currentFile, setCurrentFile] = useState("");
    const [error, setError] = useState("");
    const [downloadUrl, setDownloadUrl] = useState("");
    const [summary, setSummary] = useState<MigrationSummary | null>(null);
    const [results, setResults] = useState<MigrationResult[]>([]);
    const [migrationId, setMigrationId] = useState("");
    const [selectedFile, setSelectedFile] = useState<MigrationResult | null>(null);

    // Poll for status updates
    useEffect(() => {
        if (!migrationId || status === "completed" || status === "failed") return;

        const interval = setInterval(async () => {
            try {
                const res = await axios.get(`${API_URL}/batch/status/${migrationId}`);
                const data = res.data;

                if (data.status) {
                    setStatus(data.status);
                    setProgress(data.progress || 0);
                    setCurrentFile(data.current_file || "");

                    if (data.status === "completed") {
                        // Fetch final results
                        const resultRes = await axios.get(`${API_URL}/batch/result/${migrationId}`);
                        setDownloadUrl(resultRes.data.download_url);
                        setSummary(resultRes.data.summary);
                        setResults(resultRes.data.files || []);
                        clearInterval(interval);
                    } else if (data.status === "failed") {
                        setError(data.error || "Migration failed");
                        clearInterval(interval);
                    }
                }
            } catch (err) {
                console.error("Status poll error:", err);
            }
        }, 1000);

        return () => clearInterval(interval);
    }, [migrationId, status]);

    const handleGitHubMigration = async () => {
        if (!repoUrl) {
            setError("Please enter a GitHub URL");
            return;
        }

        setStatus("cloning");
        setError("");
        setProgress(0);

        try {
            const res = await axios.post(`${API_URL}/batch/github`, {
                repo_url: repoUrl,
                branch: branch,
                source_framework: "flask",
                target_framework: "fastapi",
            });

            setMigrationId(res.data.migration_id);
        } catch (error) {
            setStatus("failed");
            if (axios.isAxiosError(error)) {
                setError(error.response?.data?.detail || error.message || "Migration failed");
            } else {
                setError("An unexpected error occurred");
            }
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        if (!file.name.endsWith(".zip")) {
            setError("Please upload a ZIP file");
            return;
        }

        setStatus("analyzing");
        setError("");
        setProgress(0);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await axios.post(`${API_URL}/batch/upload`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setStatus("completed");
            setDownloadUrl(res.data.download_url);
            setSummary(res.data.summary);
        } catch (error) {
            setStatus("failed");
            if (axios.isAxiosError(error)) {
                setError(error.response?.data?.detail || error.message || "Upload failed");
            } else {
                setError("An unexpected error occurred during upload");
            }
        }
    };

    const handleDownload = () => {
        if (downloadUrl) {
            window.open(`http://localhost:8000${downloadUrl}`, "_blank");
        }
    };

    const getStatusColor = () => {
        switch (status) {
            case "completed":
                return "text-green-500";
            case "failed":
                return "text-red-500";
            default:
                return "text-blue-500";
        }
    };

    const getStatusIcon = () => {
        switch (status) {
            case "completed":
                return <CheckCircle className="w-6 h-6" />;
            case "failed":
                return <XCircle className="w-6 h-6" />;
            default:
                return <Loader2 className="w-6 h-6 animate-spin" />;
        }
    };

    return (
        <div className="min-h-screen bg-[var(--background)]">
            {/* Header */}
            <header className="border-b border-[var(--border)] px-6 py-4">
                <div className="max-w-7xl mx-auto flex items-center gap-4">
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-[var(--card)] rounded-lg transition"
                    >
                        <ArrowLeft className="w-5 h-5" />
                    </button>
                    <h1 className="text-xl font-bold">Migration Dashboard</h1>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-8">
                {status === "idle" ? (
                    /* Input Selection */
                    <div className="max-w-2xl mx-auto">
                        {/* Toggle */}
                        <div className="flex gap-2 mb-8 p-1 bg-[var(--card)] rounded-lg w-fit mx-auto">
                            <button
                                onClick={() => setInputType("github")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition ${inputType === "github"
                                    ? "bg-[var(--primary)] text-white"
                                    : "text-gray-400 hover:text-white"
                                    }`}
                            >
                                <Github className="w-4 h-4" />
                                GitHub URL
                            </button>
                            <button
                                onClick={() => setInputType("upload")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition ${inputType === "upload"
                                    ? "bg-[var(--primary)] text-white"
                                    : "text-gray-400 hover:text-white"
                                    }`}
                            >
                                <Upload className="w-4 h-4" />
                                Upload ZIP
                            </button>
                        </div>

                        {inputType === "github" ? (
                            /* GitHub Input */
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium mb-2">
                                        Repository URL
                                    </label>
                                    <input
                                        type="text"
                                        value={repoUrl}
                                        onChange={(e) => setRepoUrl(e.target.value)}
                                        placeholder="https://github.com/username/flask-app"
                                        className="w-full px-4 py-3 bg-[var(--card)] border border-[var(--border)] rounded-lg focus:outline-none focus:border-[var(--primary)]"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-2">
                                        Branch
                                    </label>
                                    <input
                                        type="text"
                                        value={branch}
                                        onChange={(e) => setBranch(e.target.value)}
                                        placeholder="main"
                                        className="w-full px-4 py-3 bg-[var(--card)] border border-[var(--border)] rounded-lg focus:outline-none focus:border-[var(--primary)]"
                                    />
                                </div>
                                <button
                                    onClick={handleGitHubMigration}
                                    className="w-full py-3 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white font-semibold rounded-lg transition"
                                >
                                    Start Migration
                                </button>
                            </div>
                        ) : (
                            /* File Upload */
                            <div className="border-2 border-dashed border-[var(--border)] rounded-xl p-12 text-center hover:border-[var(--primary)] transition cursor-pointer">
                                <input
                                    type="file"
                                    accept=".zip"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                    id="file-upload"
                                />
                                <label htmlFor="file-upload" className="cursor-pointer">
                                    <Upload className="w-12 h-12 mx-auto mb-4 text-gray-500" />
                                    <p className="text-lg font-medium mb-2">
                                        Drop your ZIP file here
                                    </p>
                                    <p className="text-sm text-gray-500">
                                        or click to browse
                                    </p>
                                </label>
                            </div>
                        )}

                        {error && (
                            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-3 text-red-400">
                                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                                {error}
                            </div>
                        )}
                    </div>
                ) : (
                    /* Migration Progress / Results */
                    <div className="grid lg:grid-cols-3 gap-6">
                        {/* Status Panel */}
                        <div className="lg:col-span-1 space-y-6">
                            {/* Status Card */}
                            <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
                                <div className={`flex items-center gap-3 mb-4 ${getStatusColor()}`}>
                                    {getStatusIcon()}
                                    <span className="text-lg font-semibold capitalize">
                                        {status.replace("_", " ")}
                                    </span>
                                </div>

                                {status !== "completed" && status !== "failed" && (
                                    <>
                                        <div className="mb-2">
                                            <div className="flex justify-between text-sm mb-1">
                                                <span>Progress</span>
                                                <span>{progress.toFixed(0)}%</span>
                                            </div>
                                            <div className="h-2 bg-[var(--border)] rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-[var(--primary)] transition-all duration-300"
                                                    style={{ width: `${progress}%` }}
                                                />
                                            </div>
                                        </div>
                                        {currentFile && (
                                            <p className="text-sm text-gray-400 truncate">
                                                Processing: {currentFile}
                                            </p>
                                        )}
                                    </>
                                )}

                                {status === "failed" && error && (
                                    <p className="text-sm text-red-400">{error}</p>
                                )}

                                {status === "completed" && downloadUrl && (
                                    <button
                                        onClick={handleDownload}
                                        className="w-full mt-4 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition flex items-center justify-center gap-2"
                                    >
                                        <Download className="w-5 h-5" />
                                        Download FastAPI Project
                                    </button>
                                )}
                            </div>

                            {/* Summary Card */}
                            {summary && (
                                <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
                                    <h3 className="font-semibold mb-4">Migration Summary</h3>
                                    <div className="space-y-3 text-sm">
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Total Files</span>
                                            <span>{summary.total_files}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-gray-400">Files Migrated</span>
                                            <span className="text-green-500">{summary.files_migrated}</span>
                                        </div>
                                        {summary.total_chunks !== undefined && (
                                            <>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Chunks Succeeded</span>
                                                    <span className="text-green-500">
                                                        {summary.chunks_succeeded}
                                                    </span>
                                                </div>
                                                <div className="flex justify-between">
                                                    <span className="text-gray-400">Chunks Failed</span>
                                                    <span className="text-red-500">
                                                        {summary.chunks_failed}
                                                    </span>
                                                </div>
                                            </>
                                        )}
                                        {summary.average_confidence !== undefined && (
                                            <div className="flex justify-between">
                                                <span className="text-gray-400">Avg Confidence</span>
                                                <span>
                                                    {(summary.average_confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Files List */}
                            {results.length > 0 && (
                                <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
                                    <h3 className="font-semibold mb-4">Migrated Files</h3>
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {results.map((file, i) => (
                                            <button
                                                key={i}
                                                onClick={() => setSelectedFile(file)}
                                                className={`w-full text-left p-3 rounded-lg transition flex items-center gap-3 ${selectedFile?.source_path === file.source_path
                                                    ? "bg-[var(--primary)]/20 border border-[var(--primary)]"
                                                    : "bg-[var(--background)] hover:bg-[var(--card-hover)]"
                                                    }`}
                                            >
                                                <FileCode className="w-4 h-4 flex-shrink-0 text-gray-500" />
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm truncate">{file.source_path}</p>
                                                    <p className="text-xs text-gray-500">
                                                        {file.chunks_migrated} chunks •{" "}
                                                        {(file.confidence * 100).toFixed(0)}% confidence
                                                    </p>
                                                </div>
                                                {file.chunks_failed > 0 ? (
                                                    <AlertCircle className="w-4 h-4 text-yellow-500" />
                                                ) : (
                                                    <CheckCircle className="w-4 h-4 text-green-500" />
                                                )}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Back Button */}
                            <button
                                onClick={() => {
                                    setStatus("idle");
                                    setError("");
                                    setDownloadUrl("");
                                    setSummary(null);
                                    setResults([]);
                                    setMigrationId("");
                                    setSelectedFile(null);
                                }}
                                className="w-full py-3 border border-[var(--border)] hover:bg-[var(--card)] rounded-lg transition"
                            >
                                Start New Migration
                            </button>
                        </div>

                        {/* Code Viewer */}
                        <div className="lg:col-span-2">
                            {selectedFile ? (
                                <CodeDiffViewer file={selectedFile} />
                            ) : (
                                <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-12 text-center h-full flex items-center justify-center">
                                    <div className="text-gray-500">
                                        <FileCode className="w-16 h-16 mx-auto mb-4 opacity-50" />
                                        <p>Select a file to view the migration diff</p>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
}