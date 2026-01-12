"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import { FileCode } from "lucide-react";

// Dynamic import for Monaco (no SSR)
const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
    ssr: false,
    loading: () => (
        <div className="h-full flex items-center justify-center bg-[var(--card)]">
            <p className="text-gray-500">Loading editor...</p>
        </div>
    ),
});

// Dynamic import for Diff Viewer
const ReactDiffViewer = dynamic(() => import("react-diff-viewer-continued"), {
    ssr: false,
    loading: () => (
        <div className="h-full flex items-center justify-center bg-[var(--card)]">
            <p className="text-gray-500">Loading diff viewer...</p>
        </div>
    ),
});

interface MigrationResult {
    source_path: string;
    output_path: string;
    source_content: string;
    migrated_content: string;
    chunks_migrated: number;
    chunks_failed: number;
    confidence: number;
}

interface CodeDiffViewerProps {
    file: MigrationResult;
}

export default function CodeDiffViewer({ file }: CodeDiffViewerProps) {
    const [viewMode, setViewMode] = useState<"split" | "unified" | "source" | "migrated">("split");

    return (
        <div className="bg-[var(--card)] border border-[var(--border)] rounded-xl overflow-hidden h-full flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
                <div className="flex items-center gap-3">
                    <FileCode className="w-5 h-5 text-gray-500" />
                    <div>
                        <p className="font-medium text-sm">{file.source_path}</p>
                        <p className="text-xs text-gray-500">
                            {file.chunks_migrated} chunks migrated •{" "}
                            {(file.confidence * 100).toFixed(0)}% confidence
                        </p>
                    </div>
                </div>

                {/* View Mode Toggle */}
                <div className="flex gap-1 p-1 bg-[var(--background)] rounded-lg">
                    {(["split", "unified", "source", "migrated"] as const).map((mode) => (
                        <button
                            key={mode}
                            onClick={() => setViewMode(mode)}
                            className={`px-3 py-1 text-xs rounded transition capitalize ${viewMode === mode
                                ? "bg-[var(--primary)] text-white"
                                : "text-gray-400 hover:text-white"
                                }`}
                        >
                            {mode}
                        </button>
                    ))}
                </div>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-hidden">
                {viewMode === "split" && (
                    <div className="h-full overflow-auto p-4">
                        <ReactDiffViewer
                            oldValue={file.source_content}
                            newValue={file.migrated_content}
                            splitView={true}
                            useDarkTheme={true}
                            leftTitle="Flask (Source)"
                            rightTitle="FastAPI (Migrated)"
                            styles={{
                                variables: {
                                    dark: {
                                        diffViewerBackground: "#1a1a1a",
                                        addedBackground: "#1e3a29",
                                        removedBackground: "#3a1e1e",
                                        wordAddedBackground: "#2d5a3d",
                                        wordRemovedBackground: "#5a2d2d",
                                        addedGutterBackground: "#1e3a29",
                                        removedGutterBackground: "#3a1e1e",
                                        gutterBackground: "#1a1a1a",
                                        gutterBackgroundDark: "#141414",
                                        codeFoldBackground: "#252525",
                                        codeFoldGutterBackground: "#252525",
                                    },
                                },
                                contentText: {
                                    fontSize: "13px",
                                    fontFamily: "monospace",
                                },
                            }}
                        />
                    </div>
                )}

                {viewMode === "unified" && (
                    <div className="h-full overflow-auto p-4">
                        <ReactDiffViewer
                            oldValue={file.source_content}
                            newValue={file.migrated_content}
                            splitView={false}
                            useDarkTheme={true}
                            styles={{
                                variables: {
                                    dark: {
                                        diffViewerBackground: "#1a1a1a",
                                        addedBackground: "#1e3a29",
                                        removedBackground: "#3a1e1e",
                                        wordAddedBackground: "#2d5a3d",
                                        wordRemovedBackground: "#5a2d2d",
                                        gutterBackground: "#1a1a1a",
                                    },
                                },
                                contentText: {
                                    fontSize: "13px",
                                    fontFamily: "monospace",
                                },
                            }}
                        />
                    </div>
                )}

                {viewMode === "source" && (
                    <MonacoEditor
                        height="100%"
                        language="python"
                        theme="vs-dark"
                        value={file.source_content}
                        options={{
                            readOnly: true,
                            minimap: { enabled: false },
                            fontSize: 13,
                            lineNumbers: "on",
                            scrollBeyondLastLine: false,
                            wordWrap: "on",
                        }}
                    />
                )}

                {viewMode === "migrated" && (
                    <MonacoEditor
                        height="100%"
                        language="python"
                        theme="vs-dark"
                        value={file.migrated_content}
                        options={{
                            readOnly: true,
                            minimap: { enabled: false },
                            fontSize: 13,
                            lineNumbers: "on",
                            scrollBeyondLastLine: false,
                            wordWrap: "on",
                        }}
                    />
                )}
            </div>
        </div>
    );
}