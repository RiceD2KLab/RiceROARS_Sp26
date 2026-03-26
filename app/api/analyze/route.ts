import { NextResponse } from "next/server";
import { spawnSync } from "node:child_process";
import { writeFile, unlink } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import type {
  RoarAnalysisResult,
  RoarFlag,
  ExtractedSections,
} from "@/lib/types/roar";

const ALLOWED_EXTENSIONS = [".pdf", ".docx"];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB per file
const MAX_FILES = 20;

const SCRIPT_PATH = path.join(
  process.cwd(),
  "scripts",
  "parse_roar_docx.py"
);

function getExtension(name: string): string {
  const i = name.lastIndexOf(".");
  return i >= 0 ? name.slice(i).toLowerCase() : "";
}

function isValidFile(file: File): { valid: boolean; error?: string } {
  const ext = getExtension(file.name);
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    return {
      valid: false,
      error: `Invalid file type: ${file.name}. Use .pdf or .docx`,
    };
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      valid: false,
      error: `File too large: ${file.name} (max 10 MB)`,
    };
  }
  return { valid: true };
}

function mockFlagsForFile(filename: string): RoarFlag[] {
  const flags: RoarFlag[] = [
    {
      id: `mock-${filename}-1`,
      category: "structural",
      code: "missing_target",
      message:
        "No clear performance target or threshold stated in methods or results.",
      section: "methods",
    },
    {
      id: `mock-${filename}-2`,
      category: "semantic",
      code: "vague_wording",
      message: "PLO or methods could be more specific and measurable.",
      section: "plo",
    },
  ];
  return flags;
}

function mockExtracted(reason: "pdf" | "docx_fallback" | "generic" = "generic"):
  | RoarAnalysisResult["extracted"]
  | undefined {
  const prefix = "(Mock Data) ";
  const sourceLabel =
    reason === "pdf"
      ? "PDF (no parser yet)"
      : reason === "docx_fallback"
      ? "ROAR parser fallback"
      : "Sample data";

  return {
    department: `${prefix}${sourceLabel}`,
    plo: `${prefix}Students will demonstrate competency in core concepts.`,
    methods: `${prefix}Direct assessment via final project rubric (1–4 scale).`,
    results_conclusions: `${prefix}85% of students scored 3 or above. Target was 80%.`,
    improvement_plan: `${prefix}Continue current approach; revisit rubric in AY 24-25.`,
  };
}

/**
 * Run the Python ROAR .docx parser on a file. Returns extracted sections
 * or null on failure (missing Python, script error, invalid docx, etc.).
 */
async function parseDocxWithScript(
  file: File,
  tempPath: string
): Promise<ExtractedSections | null> {
  const buf = Buffer.from(await file.arrayBuffer());
  await writeFile(tempPath, buf);
  try {
    const python =
      process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
    const child = spawnSync(python, [SCRIPT_PATH, tempPath], {
      encoding: "utf-8",
      timeout: 30000,
    });

    if (child.status !== 0) {
      console.error("ROAR parser failed", {
        status: child.status,
        stderr: child.stderr,
      });
      return null;
    }

    const out = child.stdout?.trim();
    if (!out) {
      console.error("ROAR parser produced no output");
      return null;
    }

    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(out) as Record<string, unknown>;
    } catch (e) {
      console.error("ROAR parser JSON parse error", e);
      return null;
    }

    if (parsed && typeof parsed === "object" && "error" in parsed) {
      console.error("ROAR parser reported error", parsed);
      return null;
    }

    return parsed as ExtractedSections;
  } catch (e) {
    console.error("ROAR parser invocation error", e);
    return null;
  } finally {
    try {
      await unlink(tempPath);
    } catch {
      // ignore cleanup errors
    }
  }
}

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const entries = Array.from(formData.entries()).filter(
      (entry): entry is [string, File] => entry[1] instanceof File
    );
    const files = entries.map(([, file]) => file);

    if (files.length === 0) {
      return NextResponse.json(
        {
          error:
            "No files provided. Upload one or more .pdf or .docx files.",
        },
        { status: 400 }
      );
    }

    if (files.length > MAX_FILES) {
      return NextResponse.json(
        {
          error: `Too many files. Maximum ${MAX_FILES} files per request.`,
        },
        { status: 400 }
      );
    }

    for (const file of files) {
      const { valid, error } = isValidFile(file);
      if (!valid) {
        return NextResponse.json({ error }, { status: 400 });
      }
    }

    const hasDocx = files.some((f) => getExtension(f.name) === ".docx");
    // Keep a small delay so the UI shows a spinner, but shorter when we're
    // doing real work with the parser.
    if (hasDocx) {
      await new Promise((r) => setTimeout(r, 200));
    } else {
      await new Promise((r) => setTimeout(r, 400));
    }

    const results: RoarAnalysisResult[] = [];

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const filename = file.name;
      const ext = getExtension(filename);

      let extracted: RoarAnalysisResult["extracted"];

      if (ext === ".docx") {
        const tempDir = os.tmpdir();
        const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
        const tempPath = path.join(
          tempDir,
          `roar-${Date.now()}-${i}-${safeName}`
        );

        const parsed = await parseDocxWithScript(file, tempPath);
        if (parsed && Object.keys(parsed).length > 0) {
          extracted = parsed;
        } else {
          // Fallback for .docx when parser fails: mark clearly as mock data.
          extracted = mockExtracted("docx_fallback");
        }
      } else {
        // PDFs still use mock extracted data.
        extracted = mockExtracted("pdf");
      }

      results.push({
        filename,
        extracted,
        // Flags and section scores are still mock for now.
        flags: mockFlagsForFile(filename),
        sectionScores: { plo: 1, methods: 1, results: 1, plan: 1 },
      });
    }

    return NextResponse.json(results);
  } catch (e) {
    console.error("POST /api/analyze error:", e);
    return NextResponse.json(
      { error: "Analysis request failed." },
      { status: 500 }
    );
  }
}

