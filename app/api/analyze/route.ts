import { NextResponse } from "next/server";
import { spawnSync } from "node:child_process";
import { writeFile, unlink } from "node:fs/promises";
import path from "node:path";
import os from "node:os";
import type {
  RoarAnalysisResult,
  RoarFlag,
  ExtractedSections,
  SectionScores,
} from "@/lib/types/roar";

const ALLOWED_EXTENSIONS = [".pdf", ".docx"];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB per file
const MAX_FILES = 20;

const PARSE_SCRIPT_PATH = path.join(
  process.cwd(),
  "scripts",
  "parse_roar_docx.py"
);

const EVAL_SCRIPT_PATH = path.join(
  process.cwd(),
  "scripts",
  "run_roar_evaluation.py"
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
      ? "PDF (pipeline unavailable)"
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
 * One flag per section that failed the binary quality score (0).
 */
function flagsFromSectionScores(filename: string, scores: SectionScores): RoarFlag[] {
  const out: RoarFlag[] = [];
  const entries: {
    key: keyof SectionScores;
    section: NonNullable<RoarFlag["section"]>;
    code: string;
    label: string;
  }[] = [
    { key: "plo", section: "plo", code: "plo_quality", label: "PLO" },
    { key: "methods", section: "methods", code: "methods_quality", label: "Methods" },
    { key: "results", section: "results", code: "results_quality", label: "Results" },
    {
      key: "plan",
      section: "improvement_plan",
      code: "plan_quality",
      label: "Improvement plan",
    },
  ];
  for (const { key, section, code, label } of entries) {
    if (scores[key] === 0) {
      out.push({
        id: `${filename}-${code}`,
        category: "semantic",
        code,
        message: `${label} section did not meet the model quality threshold.`,
        section,
      });
    }
  }
  return out;
}

type PipelineSuccess = {
  ok: true;
  extracted: ExtractedSections;
  sectionScores: SectionScores;
  weightedScore: number;
  consistent: boolean;
  iterations: number;
  strictAllPass?: boolean;
  classificationStrategy?: string;
  roarPromptSet?: string;
  roarModelProfile?: string;
  evaluatorModel?: string;
  verifierModel?: string;
};

type PipelineFailure = {
  ok: false;
  error: string;
  detail?: string;
  debugContext?: Record<string, unknown>;
};

/**
 * Run evaluation_pipeline via scripts/run_roar_evaluation.py.
 */
async function runEvaluationPipeline(
  file: File,
  tempPath: string,
  ext: ".pdf" | ".docx",
  evaluatorModel?: string,
  verifierModel?: string
): Promise<PipelineSuccess | PipelineFailure> {
  const buf = Buffer.from(await file.arrayBuffer());
  await writeFile(tempPath, buf);

  try {
    const python =
      process.env.PYTHON || (process.platform === "win32" ? "python" : "python3");
    const args =
      ext === ".pdf"
        ? [EVAL_SCRIPT_PATH, "--pdf", tempPath]
        : [EVAL_SCRIPT_PATH, "--docx", tempPath];
    if (evaluatorModel) {
      args.push("--evaluator-model", evaluatorModel);
    }
    if (verifierModel) {
      args.push("--verifier-model", verifierModel);
    }
    const timeout = Number(process.env.ROAR_PIPELINE_TIMEOUT_MS || "300000");
    const child = spawnSync(python, args, {
      encoding: "utf-8",
      timeout,
      maxBuffer: 10 * 1024 * 1024,
    });

    const out = child.stdout?.trim() ?? "";
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(out) as Record<string, unknown>;
    } catch {
      const stderr = child.stderr?.trim().slice(0, 800) ?? "";
      return {
        ok: false,
        error: `Invalid JSON from pipeline (exit ${child.status}). ${stderr}`,
      };
    }

    if (parsed.ok === false) {
      return {
        ok: false,
        error: String(parsed.error ?? "Pipeline reported failure"),
        detail:
          typeof parsed.detail === "string" ? parsed.detail.slice(0, 1200) : undefined,
        debugContext:
          parsed.debugContext && typeof parsed.debugContext === "object"
            ? (parsed.debugContext as Record<string, unknown>)
            : undefined,
      };
    }

    if (parsed.ok !== true) {
      return { ok: false, error: "Unexpected pipeline response shape" };
    }

    const extracted = parsed.extracted as ExtractedSections | undefined;
    const sectionScores = parsed.sectionScores as SectionScores | undefined;
    if (!extracted || !sectionScores) {
      return { ok: false, error: "Pipeline response missing extracted or scores" };
    }

    return {
      ok: true,
      extracted,
      sectionScores,
      weightedScore: Number(parsed.weightedScore),
      consistent: Boolean(parsed.consistent),
      iterations: Number(parsed.iterations),
      strictAllPass:
        typeof parsed.strictAllPass === "boolean"
          ? parsed.strictAllPass
          : undefined,
      classificationStrategy:
        typeof parsed.classificationStrategy === "string"
          ? parsed.classificationStrategy
          : undefined,
      roarPromptSet:
        typeof parsed.roarPromptSet === "string"
          ? parsed.roarPromptSet
          : undefined,
      roarModelProfile:
        typeof parsed.roarModelProfile === "string"
          ? parsed.roarModelProfile
          : undefined,
      evaluatorModel:
        typeof parsed.evaluatorModel === "string" ? parsed.evaluatorModel : undefined,
      verifierModel:
        typeof parsed.verifierModel === "string" ? parsed.verifierModel : undefined,
    };
  } finally {
    try {
      await unlink(tempPath);
    } catch {
      // ignore cleanup errors
    }
  }
}

/**
 * Run the legacy Python ROAR .docx parser. Returns extracted sections
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
    const child = spawnSync(python, [PARSE_SCRIPT_PATH, tempPath], {
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
    const evaluatorModel = formData.get("evaluatorModel");
    const verifierModel = formData.get("verifierModel");
    const evaluatorModelValue =
      typeof evaluatorModel === "string" ? evaluatorModel.trim() : "";
    const verifierModelValue =
      typeof verifierModel === "string" ? verifierModel.trim() : "";

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

      const tempDir = os.tmpdir();
      const safeName = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
      const tempPath = path.join(
        tempDir,
        `roar-${Date.now()}-${i}-${safeName}`
      );

      const pipeline = await runEvaluationPipeline(
        file,
        tempPath,
        ext === ".docx" ? ".docx" : ".pdf",
        evaluatorModelValue || undefined,
        verifierModelValue || undefined
      );

      if (pipeline.ok) {
        results.push({
          filename,
          extracted: pipeline.extracted,
          flags: flagsFromSectionScores(filename, pipeline.sectionScores),
          sectionScores: pipeline.sectionScores,
          weightedScore: pipeline.weightedScore,
          pipelineConsistent: pipeline.consistent,
          pipelineIterations: pipeline.iterations,
          strictAllPass: pipeline.strictAllPass,
          classificationStrategy: pipeline.classificationStrategy as
            | "strict"
            | "weighted"
            | undefined,
          roarPromptSet: pipeline.roarPromptSet,
          roarModelProfile: pipeline.roarModelProfile,
          evaluatorModel: pipeline.evaluatorModel,
          verifierModel: pipeline.verifierModel,
        });
        continue;
      }

      console.error("ROAR evaluation pipeline failed:", pipeline.error, {
        detail: pipeline.detail,
        debugContext: pipeline.debugContext,
      });

      let extracted: RoarAnalysisResult["extracted"];
      let analysisNote = `Evaluation pipeline unavailable: ${pipeline.error}`;
      if (pipeline.debugContext) {
        const base = pipeline.debugContext.evaluatorApiBase;
        const ver = pipeline.debugContext.evaluatorApiVersion;
        if (typeof base === "string" || typeof ver === "string") {
          analysisNote += ` (evaluator base=${base || "unset"}, api-version=${ver || "unset"})`;
        }
      }

      if (ext === ".docx") {
        const fallbackPath = path.join(
          tempDir,
          `roar-parse-${Date.now()}-${i}-${safeName}`
        );
        const parsed = await parseDocxWithScript(file, fallbackPath);
        if (parsed && Object.keys(parsed).length > 0) {
          extracted = parsed;
          analysisNote +=
            " Showing sections from the legacy .docx parser; scores not computed.";
        } else {
          extracted = mockExtracted("docx_fallback");
          analysisNote +=
            " Legacy parser also failed; showing placeholder extracted text.";
        }
      } else {
        extracted = mockExtracted("pdf");
        analysisNote +=
          " PDF text extraction or scoring failed; showing placeholder content.";
      }

      results.push({
        filename,
        extracted,
        flags: mockFlagsForFile(filename),
        analysisNote,
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
