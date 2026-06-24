"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
    Circle,
    MinusCircle,
} from "lucide-react";
import axios from "axios";
import CodeDiffViewer from "./CodeDiffViewer";

// Configurable so the UI can point at a non-localhost backend in deployment.
const API_ORIGIN = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_URL = `${API_ORIGIN}/api/v1`;

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

type StepState = "pending" | "active" | "done" | "failed" | "skipped";

interface Step {
    key: string;
    label: string;
    state: StepState;
    detail?: string;
}

// queued/running map to the in-progress UI; completed/failed are terminal.
type Status = "idle" | "queued" | "running" | "completed" | "failed";

export default function MigrationPanel({ onBack }: { onBack: () => void }) {
    const [inputType, setInputType] = useState<"github" | "upload">("github");
    const [repoUrl, setRepoUrl] = useState("");
    const [branch, setBranch] = useState("main");
    const [status, setStatus] = useState<Status>("idle");
    const [progress, setProgress] = useState(0);
    const [steps, setSteps] = useState<Step[]>([]);
    const [currentFile, setCurrentFile] = useState("");
    const [error, setError] = useState("");
    const [downloadUrl, setDownloadUrl] = useState("");
    const [summary, setSummary] = useState<MigrationSummary | null>(null);
    const [results, setResults] = useState<MigrationResult[]>([]);
    const [migrationId, setMigrationId] = useState("");
    const [selectedFile, setSelectedFile] = useState<MigrationResult | null>(null);

    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const stopPolling = useCallback(() => {
        if (pollRef.current) {
            clearInterval(pollRef.current);
            pollRef.current = null;
        }
    }, []);

    const resetState = useCallback(() => {
        stopPolling();
        setStatus("idle");
        setProgress(0);
        setSteps([]);
        setCurrentFile("");
        setError("");
        setDownloadUrl("");
        setSummary(null);
        setResults([]);
        setMigrationId("");
        setSelectedFile(null);
    }, [stopPolling]);

    // Single polling loop shared by both the GitHub and upload flows.
    useEffect(() => {
        if (!migrationId) return;
        if (status === "completed" || status === "failed") return;

        const tick = async () => {
            try {
                const { data } = await axios.get(`${API_URL}/batch/status/${migrationId}`);
                setStatus(data.status);
                setProgress(data.progress ?? 0);
                setSteps(data.steps ?? []);
                setCurrentFile(data.current_file ?? "");

                if (data.status === "completed") {
                    stopPolling();
                    const { data: result } = await axios.get(`${API_URL}/batch/result/${migrationId}`);
                    setDownloadUrl(result.download_url ?? "");
                    setSummary(result.summary ?? null);
                    setResults(result.files ?? []);
                } else if (data.status === "failed") {
                    stopPolling();
                    setError((data.errors && data.errors[0]) || "Migration failed");
                }
            } catch (err) {
                console.error("Status poll error:", err);
            }
        };

        pollRef.current = setInterval(tick, 1000);
        tick(); // fire immediately so the UI doesn't wait a full second
        return stopPolling;
    }, [migrationId, status, stopPolling]);

    const handleGitHubMigration = async () => {
        if (!repoUrl) {
            setError("Please enter a GitHub URL");
            return;
        }
        setError("");
        setProgress(0);
        setStatus("queued");
        try {
            const { data } = await axios.post(`${API_URL}/batch/github`, {
                repo_url: repoUrl,
                branch,
                source_framework: "flask",
                target_framework: "fastapi",
            });
            setMigrationId(data.migration_id);
        } catch (err) {
            setStatus("failed");
            setError(extractError(err, "Migration failed"));
        }
    };

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".zip")) {
            setError("Please upload a ZIP file");
            return;
        }
        setError("");
        setProgress(0);
        setStatus("queued");

        const formData = new FormData();
        formData.append("file", file);
        try {
            const { data } = await axios.post(`${API_URL}/batch/upload`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            // Upload now runs in the background and returns an id, exactly like
            // the GitHub flow, so the same polling loop drives the UI.
            setMigrationId(data.migration_id);
        } catch (err) {
            setStatus("failed");
            setError(extractError(err, "Upload failed"));
        }
    };

    const handleDownload = () => {
        if (downloadUrl) window.open(`${API_ORIGIN}${downloadUrl}`, "_blank");
    };

    const inProgress = status === "queued" || status === "running";

    return (
        <div className="min-h-screen bg-[var(--background)]">
            <header className="border-b border-[var(--border)] px-6 py-4">
                <div className="max-w-7xl mx-auto flex items-center gap-4">
                    <button onClick={onBack} className="p-2 hover:bg-[var(--card)] rounded-lg transition">
                        <ArrowLeft className="w-5 h-5" />
                    </button>
                    <h1 className="text-xl font-bold">Migration Dashboard</h1>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-6 py-8">
                {status === "idle" ? (
                    <div className="max-w-2xl mx-auto">
                        <div className="flex gap-2 mb-8 p-1 bg-[var(--card)] rounded-lg w-fit mx-auto">
                            <button
                                onClick={() => setInputType("github")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition ${inputType === "github" ? "bg-[var(--primary)] text-white" : "text-gray-400 hover:text-white"}`}
                            >
                                <Github className="w-4 h-4" /> GitHub URL
                            </button>
                            <button
                                onClick={() => setInputType("upload")}
                                className={`flex items-center gap-2 px-4 py-2 rounded-md transition ${inputType === "upload" ? "bg-[var(--primary)] text-white" : "text-gray-400 hover:text-white"}`}
                            >
                                <Upload className="w-4 h-4" /> Upload ZIP
                            </button>
                        </div>

                        {inputType === "github" ? (
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-sm font-medium mb-2">Repository URL</label>
                                    <input
                                        type="text"
                                        value={repoUrl}
                                        onChange={(e) => setRepoUrl(e.target.value)}
                                        placeholder="https://github.com/username/flask-app"
                                        className="w-full px-4 py-3 bg-[var(--card)] border border-[var(--border)] rounded-lg focus:outline-none focus:border-[var(--primary)]"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-medium mb-2">Branch</label>
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
                            <div className="border-2 border-dashed border-[var(--border)] rounded-xl p-12 text-center hover:border-[var(--primary)] transition cursor-pointer">
                                <input type="file" accept=".zip" onChange={handleFileUpload} className="hidden" id="file-upload" />
                                <label htmlFor="file-upload" className="cursor-pointer">
                                    <Upload className="w-12 h-12 mx-auto mb-4 text-gray-500" />
                                    <p className="text-lg font-medium mb-2">Drop your ZIP file here</p>
                                    <p className="text-sm text-gray-500">or click to browse</p>
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
                    <div className="grid lg:grid-cols-3 gap-6">
                        <div className="lg:col-span-1 space-y-6">
                            {/* Status + steps */}
                            <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
                                <div className={`flex items-center gap-3 mb-4 ${status === "completed" ? "text-green-500" : status === "failed" ? "text-red-500" : "text-blue-500"}`}>
                                    {status === "completed" ? <CheckCircle className="w-6 h-6" /> : status === "failed" ? <XCircle className="w-6 h-6" /> : <Loader2 className="w-6 h-6 animate-spin" />}
                                    <span className="text-lg font-semibold capitalize">{status}</span>
                                </div>

                                {inProgress && (
                                    <div className="mb-4">
                                        <div className="flex justify-between text-sm mb-1">
                                            <span>Progress</span>
                                            <span>{progress.toFixed(0)}%</span>
                                        </div>
                                        <div className="h-2 bg-[var(--border)] rounded-full overflow-hidden">
                                            <div className="h-full bg-[var(--primary)] transition-all duration-300" style={{ width: `${progress}%` }} />
                                        </div>
                                    </div>
                                )}

                                {/* Step checklist */}
                                {steps.length > 0 && (
                                    <ol className="space-y-3">
                                        {steps.map((step) => (
                                            <li key={step.key} className="flex items-start gap-3">
                                                <StepIcon state={step.state} />
                                                <div className="min-w-0">
                                                    <p className={`text-sm ${step.state === "active" ? "font-semibold text-[var(--foreground)]" : step.state === "pending" ? "text-gray-500" : ""}`}>
                                                        {step.label}
                                                    </p>
                                                    {step.state === "active" && currentFile && step.key === "migrate" && (
                                                        <p className="text-xs text-gray-400 truncate">{currentFile}</p>
                                                    )}
                                                    {step.detail && step.state !== "active" && (
                                                        <p className="text-xs text-gray-500 truncate">{step.detail}</p>
                                                    )}
                                                </div>
                                            </li>
                                        ))}
                                    </ol>
                                )}

                                {status === "failed" && error && (
                                    <p className="mt-4 text-sm text-red-400">{error}</p>
                                )}

                                {status === "completed" && downloadUrl && (
                                    <button
                                        onClick={handleDownload}
                                        className="w-full mt-4 py-3 bg-green-600 hover:bg-green-700 text-white font-semibold rounded-lg transition flex items-center justify-center gap-2"
                                    >
                                        <Download className="w-5 h-5" /> Download FastAPI Project
                                    </button>
                                )}
                            </div>

                            {summary && (
                                <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
                                    <h3 className="font-semibold mb-4">Migration Summary</h3>
                                    <div className="space-y-3 text-sm">
                                        <Row label="Total Files" value={`${summary.total_files}`} />
                                        <Row label="Files Migrated" value={`${summary.files_migrated}`} valueClass="text-green-500" />
                                        {summary.total_chunks !== undefined && (
                                            <>
                                                <Row label="Chunks Succeeded" value={`${summary.chunks_succeeded}`} valueClass="text-green-500" />
                                                <Row label="Chunks Failed" value={`${summary.chunks_failed}`} valueClass="text-red-500" />
                                            </>
                                        )}
                                        {summary.average_confidence !== undefined && (
                                            <Row label="Avg Confidence" value={`${(summary.average_confidence * 100).toFixed(0)}%`} />
                                        )}
                                    </div>
                                </div>
                            )}

                            {results.length > 0 && (
                                <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl p-6">
                                    <h3 className="font-semibold mb-4">Migrated Files</h3>
                                    <div className="space-y-2 max-h-64 overflow-y-auto">
                                        {results.map((file, i) => (
                                            <button
                                                key={i}
                                                onClick={() => setSelectedFile(file)}
                                                className={`w-full text-left p-3 rounded-lg transition flex items-center gap-3 ${selectedFile?.source_path === file.source_path ? "bg-[var(--primary)]/20 border border-[var(--primary)]" : "bg-[var(--background)] hover:bg-[var(--card-hover)]"}`}
                                            >
                                                <FileCode className="w-4 h-4 flex-shrink-0 text-gray-500" />
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-sm truncate">{file.source_path}</p>
                                                    <p className="text-xs text-gray-500">
                                                        {file.chunks_migrated} chunks • {(file.confidence * 100).toFixed(0)}% confidence
                                                    </p>
                                                </div>
                                                {file.chunks_failed > 0 ? <AlertCircle className="w-4 h-4 text-yellow-500" /> : <CheckCircle className="w-4 h-4 text-green-500" />}
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <button
                                onClick={resetState}
                                className="w-full py-3 border border-[var(--border)] hover:bg-[var(--card)] rounded-lg transition"
                            >
                                Start New Migration
                            </button>
                        </div>

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

function StepIcon({ state }: { state: StepState }) {
    if (state === "done") return <CheckCircle className="w-5 h-5 flex-shrink-0 text-green-500" />;
    if (state === "failed") return <XCircle className="w-5 h-5 flex-shrink-0 text-red-500" />;
    if (state === "active") return <Loader2 className="w-5 h-5 flex-shrink-0 text-blue-500 animate-spin" />;
    if (state === "skipped") return <MinusCircle className="w-5 h-5 flex-shrink-0 text-gray-600" />;
    return <Circle className="w-5 h-5 flex-shrink-0 text-gray-600" />;
}

function Row({ label, value, valueClass = "" }: { label: string; value: string; valueClass?: string }) {
    return (
        <div className="flex justify-between">
            <span className="text-gray-400">{label}</span>
            <span className={valueClass}>{value}</span>
        </div>
    );
}

function extractError(err: unknown, fallback: string): string {
    if (axios.isAxiosError(err)) {
        return err.response?.data?.detail || err.message || fallback;
    }
    return fallback;
}
