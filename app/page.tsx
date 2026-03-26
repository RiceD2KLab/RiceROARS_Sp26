"use client";

import { useState, useCallback } from "react";
import type { RoarAnalysisResult } from "@/lib/types/roar";

const ACCEPT = ".pdf,.docx";
const MAX_FILES = 20;

const FILE_COLUMN_MIN_WIDTH = 320;
const FLAGS_COLUMN_WIDTH = 280;

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [results, setResults] = useState<RoarAnalysisResult[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const selectedResult =
    results?.find((r) => r.filename === selectedFile) ?? null;

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files;
    if (!selected?.length) return;
    const list = Array.from(selected).slice(0, MAX_FILES);
    setFiles(list);
    setResults(null);
    setError(null);
    setSelectedFile(null);
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const items = e.dataTransfer.files;
      if (!items?.length) return;
      const list = Array.from(items).slice(0, MAX_FILES);
      setFiles(list);
      setResults(null);
      setError(null);
      setSelectedFile(null);
    },
    []
  );

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  }, []);

  const analyze = async () => {
    if (!files.length) {
      setError("Please select one or more .pdf or .docx files.");
      return;
    }
    setError(null);
    setLoading(true);
    setResults(null);
    setSelectedFile(null);
    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("files", f));
      const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error ?? "Analysis request failed.");
        return;
      }
      setResults(Array.isArray(data) ? data : [data]);
    } catch {
      setError("Analysis request failed.");
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setFiles([]);
    setResults(null);
    setError(null);
    setSelectedFile(null);
  };

  return (
    <div className="min-h-screen bg-white text-zinc-900 flex flex-col">
      <div className="flex flex-1 min-h-0">
        {/* Left panel: title, upload, table — 45% width */}
        <div
          className="flex flex-col w-[45%] min-w-0 border-r border-zinc-200"
          style={{ minWidth: 320 }}
        >
          <header className="shrink-0 px-4 py-4 border-b border-zinc-200">
            <h1 className="text-base font-semibold tracking-tight text-zinc-900">
              ROARS — Rice Outcomes Assessment Reporting Screening
            </h1>
            <p className="mt-1 text-xs text-zinc-600">
              Upload .pdf or .docx to get flags. Reviewer decision-support only.
            </p>
          </header>

          <div className="shrink-0 px-4 py-3 space-y-3">
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              className="rounded-lg border-2 border-dashed border-zinc-300 bg-zinc-50/80 p-4 text-center"
            >
              <input
                type="file"
                accept={ACCEPT}
                multiple
                onChange={onFileChange}
                className="sr-only"
                id="roar-file-input"
              />
              <label
                htmlFor="roar-file-input"
                className="cursor-pointer text-xs text-zinc-600 hover:text-zinc-900"
              >
                Drag and drop or click to browse
              </label>
              <p className="mt-0.5 text-xs text-zinc-500">
                .pdf, .docx (max {MAX_FILES} files, 10 MB each)
              </p>
              {files.length > 0 && (
                <p className="mt-1 text-xs font-medium text-zinc-700">
                  {files.length} file{files.length !== 1 ? "s" : ""} selected
                </p>
              )}
            </div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={analyze}
                disabled={loading || files.length === 0}
                className="rounded-lg bg-zinc-900 text-white px-3 py-1.5 text-sm font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-800"
              >
                {loading ? "Analyzing…" : "Analyze"}
              </button>
              {(files.length > 0 || results) && (
                <button
                  type="button"
                  onClick={clear}
                  disabled={loading}
                  className="rounded-lg border border-zinc-300 px-3 py-1.5 text-sm font-medium text-zinc-700 hover:bg-zinc-100 disabled:opacity-50"
                >
                  Clear
                </button>
              )}
            </div>
            {error && (
              <div className="rounded-lg bg-red-50 border border-red-200 text-red-800 px-3 py-2 text-xs">
                {error}
              </div>
            )}
          </div>

          {/* Table: fixed column widths, horizontally scrollable */}
          {results && results.length > 0 && (
            <div className="flex-1 min-h-0 flex flex-col border-t border-zinc-200">
              <h2 className="shrink-0 px-4 py-2 text-sm font-semibold text-zinc-900">
                Results
              </h2>
              <div className="flex-1 min-h-0 overflow-x-auto overflow-y-auto">
                <table
                  className="w-full text-sm"
                  style={{
                    minWidth: FILE_COLUMN_MIN_WIDTH + FLAGS_COLUMN_WIDTH,
                  }}
                >
                  <thead className="sticky top-0 bg-zinc-50 z-10">
                    <tr className="border-b border-zinc-200">
                      <th
                        className="px-3 py-2 text-left font-medium text-zinc-700 whitespace-nowrap"
                        style={{ minWidth: FILE_COLUMN_MIN_WIDTH }}
                      >
                        File
                      </th>
                      <th
                        className="px-3 py-2 text-left font-medium text-zinc-700 whitespace-nowrap"
                        style={{ minWidth: FLAGS_COLUMN_WIDTH }}
                      >
                        Flags
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((result) => (
                      <tr
                        key={result.filename}
                        onClick={() => setSelectedFile(result.filename)}
                        className={`border-b border-zinc-100 cursor-pointer ${
                          selectedFile === result.filename
                            ? "bg-blue-50"
                            : "hover:bg-zinc-50/80"
                        }`}
                      >
                        <td
                          className="px-3 py-2 font-medium text-zinc-900 whitespace-nowrap"
                          style={{ minWidth: FILE_COLUMN_MIN_WIDTH }}
                        >
                          {result.filename}
                        </td>
                        <td
                          className="px-3 py-2 align-top whitespace-nowrap"
                          style={{ minWidth: FLAGS_COLUMN_WIDTH }}
                        >
                          <div className="flex flex-wrap items-center gap-1.5">
                            <span
                              className={`inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                                result.flags.length === 0
                                  ? "bg-emerald-100 text-emerald-800"
                                  : "bg-zinc-200 text-zinc-800"
                              }`}
                            >
                              {result.flags.length}
                            </span>
                            {result.flags.length === 0 ? (
                              <span className="text-xs text-emerald-700">
                                No issues
                              </span>
                            ) : (
                              result.flags.map((f) => (
                                <span
                                  key={f.id}
                                  className="inline-flex rounded bg-zinc-200 px-1.5 py-0.5 text-xs font-medium text-zinc-700"
                                >
                                  {f.code}
                                </span>
                              ))
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Right panel: header (fixed), details below — 55% width */}
        <div className="flex flex-col flex-1 min-w-0 bg-zinc-50/50">
          <div className="shrink-0 px-4 py-4 border-b border-zinc-200">
            <h2 className="text-base font-semibold tracking-tight text-zinc-900">
              Section scores
            </h2>
            <p className="mt-1 text-xs text-zinc-600">
              Select a row to view section scores and details.
            </p>
          </div>

          <div className="flex-1 min-h-0 overflow-auto px-4 py-3">
            <h2 className="text-sm font-semibold text-zinc-900 mb-2">
              Details
            </h2>
            {selectedResult ? (
              <div className="space-y-4 text-sm">
                {selectedResult.sectionScores && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2">
                      Section scores
                    </h3>
                    <div className="flex flex-wrap gap-4 text-zinc-600">
                      <span>
                        PLO:{" "}
                        <span className="font-medium text-zinc-900">
                          {selectedResult.sectionScores.plo}
                        </span>
                      </span>
                      <span>
                        Methods:{" "}
                        <span className="font-medium text-zinc-900">
                          {selectedResult.sectionScores.methods}
                        </span>
                      </span>
                      <span>
                        Results:{" "}
                        <span className="font-medium text-zinc-900">
                          {selectedResult.sectionScores.results}
                        </span>
                      </span>
                      <span>
                        Plan:{" "}
                        <span className="font-medium text-zinc-900">
                          {selectedResult.sectionScores.plan}
                        </span>
                      </span>
                    </div>
                  </div>
                )}
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2">
                    Flags
                  </h3>
                  {selectedResult.flags.length === 0 ? (
                    <p className="text-zinc-600">
                      No issues found for this ROAR.
                    </p>
                  ) : (
                    <ul className="space-y-1.5">
                      {selectedResult.flags.map((flag) => (
                        <li
                          key={flag.id}
                          className="flex flex-wrap gap-2 items-start"
                        >
                          <span
                            className={
                              flag.category === "structural"
                                ? "rounded bg-amber-100 px-1.5 py-0.5 text-xs font-medium text-amber-800"
                                : "rounded bg-blue-100 px-1.5 py-0.5 text-xs font-medium text-blue-800"
                            }
                          >
                            {flag.category}
                          </span>
                          {flag.section && (
                            <span className="text-zinc-500">{flag.section}</span>
                          )}
                          <span className="text-zinc-700">{flag.message}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
                {selectedResult.extracted && (
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 mb-2">
                      Extracted sections
                    </h3>
                    <dl className="space-y-1.5">
                      {selectedResult.extracted.department && (
                        <>
                          <dt className="font-medium text-zinc-500">
                            Department
                          </dt>
                          <dd className="text-zinc-700 pl-0">
                            {selectedResult.extracted.department}
                          </dd>
                        </>
                      )}
                      {selectedResult.extracted.plo && (
                        <>
                          <dt className="font-medium text-zinc-500">PLO</dt>
                          <dd className="text-zinc-700 pl-0">
                            {selectedResult.extracted.plo}
                          </dd>
                        </>
                      )}
                      {selectedResult.extracted.methods && (
                        <>
                          <dt className="font-medium text-zinc-500">
                            Methods
                          </dt>
                          <dd className="text-zinc-700 pl-0">
                            {selectedResult.extracted.methods}
                          </dd>
                        </>
                      )}
                      {selectedResult.extracted.results_conclusions && (
                        <>
                          <dt className="font-medium text-zinc-500">
                            Results
                          </dt>
                          <dd className="text-zinc-700 pl-0">
                            {
                              selectedResult.extracted.results_conclusions
                            }
                          </dd>
                        </>
                      )}
                      {selectedResult.extracted.improvement_plan && (
                        <>
                          <dt className="font-medium text-zinc-500">
                            Improvement plan
                          </dt>
                          <dd className="text-zinc-700 pl-0">
                            {
                              selectedResult.extracted.improvement_plan
                            }
                          </dd>
                        </>
                      )}
                    </dl>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-zinc-500">
                Click a row in the table to view details here.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
