/**
 * Index Health Tool
 *
 * Returns a health dashboard for all vector database tables:
 * - per-table row count, last-indexed date, and disk size
 * - embedding provider and model configuration
 */

import * as fs from "fs";
import * as path from "path";
import { emptySchema, type EmptyInput } from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";

/** Human-readable display names for the known tables */
const TABLE_DISPLAY_NAMES: Record<string, string> = {
  hytale_methods: "Server Code (Methods)",
  hytale_client_ui: "Client UI",
  hytale_gamedata: "Game Data",
  hytale_docs: "Documentation",
};

/** Default embedding models per provider (mirrors embedder.ts constants) */
const PROVIDER_DEFAULT_MODELS: Record<string, { code: string; text: string }> = {
  voyage: { code: "voyage-code-3", text: "voyage-4-large" },
  ollama: { code: "nomic-embed-text", text: "nomic-embed-text" },
  openai: { code: "text-embedding-3-small", text: "text-embedding-3-small" },
  cohere: { code: "embed-english-v3.0", text: "embed-english-v3.0" },
};

/** Health info for a single table */
export interface TableHealth {
  /** Table name as stored in config (e.g., "hytale_methods") */
  name: string;
  /** Human-readable label */
  displayName: string;
  /** Whether the table exists in the vector store */
  exists: boolean;
  /** Total row count (null if table does not exist or stats failed) */
  rowCount: number | null;
  /** ISO-8601 timestamp of the last ingest operation (null if unknown) */
  lastIndexed: string | null;
  /** Total on-disk size of the table directory in bytes (null if unknown) */
  sizeBytes: number | null;
}

/** Full index health report */
export interface IndexHealthData {
  /** Per-table health information */
  tables: TableHealth[];
  /** Configured embedding provider name */
  embeddingProvider: string;
  /** Resolved model used for code embeddings */
  codeModel: string;
  /** Resolved model used for text/doc embeddings */
  textModel: string;
  /** Vector dimensions for code embeddings (null if embedding not configured) */
  codeDimensions: number | null;
  /** Vector dimensions for text embeddings (null if embedding not configured) */
  textDimensions: number | null;
  /** Absolute path to the database directory on disk (null if not set) */
  databasePath: string | null;
  /** Whether every table exists and has at least one row */
  overallHealthy: boolean;
}

/**
 * Recursively sum the size of all files under a directory.
 * Returns null if the path does not exist or cannot be read.
 */
function getDirectorySize(dirPath: string): number | null {
  try {
    let total = 0;
    const stack: string[] = [dirPath];
    while (stack.length > 0) {
      const current = stack.pop()!;
      const entries = fs.readdirSync(current, { withFileTypes: true });
      for (const entry of entries) {
        const fullPath = path.join(current, entry.name);
        if (entry.isDirectory()) {
          stack.push(fullPath);
        } else {
          try {
            total += fs.statSync(fullPath).size;
          } catch {
            // ignore unreadable files
          }
        }
      }
    }
    return total;
  } catch {
    return null;
  }
}

/**
 * Derive the last-indexed timestamp for a LanceDB table from the
 * modification time of the newest manifest file in its _versions/ directory.
 */
function getTableLastIndexed(tablePath: string): Date | null {
  const versionsPath = path.join(tablePath, "_versions");
  try {
    const files = fs.readdirSync(versionsPath).filter((f) => f.endsWith(".manifest"));
    if (files.length === 0) return null;
    const mtimes = files.map((f) => fs.statSync(path.join(versionsPath, f)).mtime.getTime());
    return new Date(Math.max(...mtimes));
  } catch {
    return null;
  }
}

/** Tool definition */
export const indexHealthTool: ToolDefinition<EmptyInput, IndexHealthData> = {
  name: "hytale_index_health",
  description:
    "Get a health dashboard for the Hytale RAG index. " +
    "Shows per-table row counts, last-indexed dates, disk sizes, " +
    "and the current embedding model configuration. " +
    "Use this to check whether the index is complete or needs re-indexing.",
  inputSchema: emptySchema,

  async handler(_input, context): Promise<ToolResult<IndexHealthData>> {
    const { config, vectorStore, embedding } = context;
    const dbPath = config.vectorStore.path ?? null;

    // Build the ordered list of tables from config
    const tableEntries: Array<{ name: string }> = [
      { name: config.tables.code },
      { name: config.tables.clientUI },
      { name: config.tables.gamedata },
      { name: config.tables.docs },
    ];

    const tables: TableHealth[] = [];

    for (const { name } of tableEntries) {
      let exists = false;
      let rowCount: number | null = null;
      let lastIndexed: string | null = null;
      let sizeBytes: number | null = null;

      try {
        exists = await vectorStore.tableExists(name);
      } catch {
        exists = false;
      }

      if (exists) {
        try {
          const stats = await vectorStore.getStats(name);
          rowCount = stats.rowCount;
        } catch {
          rowCount = null;
        }
      }

      // Filesystem-based stats: only meaningful for LanceDB with a local path
      if (dbPath && vectorStore.name === "lancedb") {
        const tableDirPath = path.join(dbPath, `${name}.lance`);
        if (fs.existsSync(tableDirPath)) {
          const lastDate = getTableLastIndexed(tableDirPath);
          lastIndexed = lastDate ? lastDate.toISOString() : null;
          sizeBytes = getDirectorySize(tableDirPath);
        }
      }

      tables.push({
        name,
        displayName: TABLE_DISPLAY_NAMES[name] ?? name,
        exists,
        rowCount,
        lastIndexed,
        sizeBytes,
      });
    }

    // Resolve embedding model names (config override → provider default)
    const provider = config.embedding.provider;
    const defaults = PROVIDER_DEFAULT_MODELS[provider] ?? { code: "unknown", text: "unknown" };
    const codeModel = config.embedding.models?.code ?? defaults.code;
    const textModel = config.embedding.models?.text ?? defaults.text;

    const codeDimensions = embedding ? embedding.getDimensions("code") : null;
    const textDimensions = embedding ? embedding.getDimensions("text") : null;

    const overallHealthy = tables.every((t) => t.exists && t.rowCount !== 0);

    return {
      success: true,
      data: {
        tables,
        embeddingProvider: provider,
        codeModel,
        textModel,
        codeDimensions,
        textDimensions,
        databasePath: dbPath,
        overallHealthy,
      },
    };
  },
};

// ─── Formatting helpers ───────────────────────────────────────────────────────

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

/** Render the health report as Markdown (used by MCP and REST servers) */
export function formatIndexHealth(data: IndexHealthData): string {
  const lines: string[] = [
    "# Hytale RAG Index Health",
    "",
    data.overallHealthy
      ? "**Overall Status:** All tables indexed and healthy"
      : "**Overall Status:** One or more tables are missing or empty",
    "",
    "## Tables",
    "",
    "| Table | Status | Rows | Last Indexed | Size |",
    "|-------|--------|------|--------------|------|",
  ];

  for (const t of data.tables) {
    const status = t.exists ? "✓ Indexed" : "✗ Missing";
    const rows = t.rowCount !== null ? t.rowCount.toLocaleString() : "—";
    const date = t.lastIndexed ? formatDate(t.lastIndexed) : "—";
    const size = t.sizeBytes !== null ? formatBytes(t.sizeBytes) : "—";
    lines.push(`| ${t.displayName} | ${status} | ${rows} | ${date} | ${size} |`);
  }

  lines.push(
    "",
    "## Embedding Configuration",
    "",
    `- **Provider:** ${data.embeddingProvider}`,
    `- **Code model:** ${data.codeModel}${data.codeDimensions !== null ? ` (${data.codeDimensions}d)` : ""}`,
    `- **Text model:** ${data.textModel}${data.textDimensions !== null ? ` (${data.textDimensions}d)` : ""}`,
  );

  if (data.databasePath) {
    lines.push(`- **Database path:** \`${data.databasePath}\``);
  }

  if (!data.overallHealthy) {
    lines.push(
      "",
      "> **Tip:** Run the ingest scripts (`npm run ingest`) to index missing tables, " +
        "or use the Hytale Toolkit Setup wizard to rebuild the database.",
    );
  }

  return lines.join("\n");
}
