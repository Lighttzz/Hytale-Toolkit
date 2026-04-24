/**
 * Cross-Collection Federated Search Tool  (search_hytale_knowledge)
 *
 * Searches all four indexes simultaneously (server code, client UI, game data,
 * documentation), then fuses the ranked lists with Reciprocal Rank Fusion (RRF)
 * so the most universally relevant results float to the top.
 *
 * When a Voyage API key is available, the fused candidates are re-ranked once
 * with voyage-reranker for a final quality boost.
 */

import { z } from "zod";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import { resolveCodePath, resolveGameDataPath } from "../../utils/paths.js";
import { rerankResults } from "../../providers/embedding/voyage-reranker.js";

// ─── Schemas ──────────────────────────────────────────────────────────────────

export const searchKnowledgeSchema = z.object({
  query: z
    .string()
    .min(1)
    .describe("Natural language question about any aspect of Hytale modding"),
  limit: z
    .number()
    .int()
    .min(1)
    .max(20)
    .optional()
    .default(8)
    .describe("Total number of fused results to return (default 8, max 20)"),
  sources: z
    .array(z.enum(["code", "client", "gamedata", "docs"]))
    .optional()
    .default(["code", "client", "gamedata", "docs"])
    .describe("Which indexes to include (default: all four)"),
});

export type SearchKnowledgeInput = z.infer<typeof searchKnowledgeSchema>;

// ─── Result type ──────────────────────────────────────────────────────────────

export interface KnowledgeResult {
  /** Which collection this result came from */
  source: "code" | "client" | "gamedata" | "docs";
  id: string;
  title: string;
  snippet: string;
  /** Link for display (resolved file path) */
  path: string;
  /** RRF-fused relevance score */
  score: number;
  /** Raw data for each source type */
  raw: Record<string, unknown>;
}

// ─── RRF helper ───────────────────────────────────────────────────────────────

/**
 * Reciprocal Rank Fusion — merges multiple ranked lists into one.
 * k = 60 is the canonical constant from the original RRF paper.
 */
function rrfMerge<T extends { id: string }>(
  rankedLists: T[][],
  k = 60
): Array<{ item: T; score: number }> {
  const scores = new Map<string, { item: T; score: number }>();

  for (const list of rankedLists) {
    for (let rank = 0; rank < list.length; rank++) {
      const item = list[rank];
      const rrf = 1 / (k + rank + 1);
      const existing = scores.get(item.id);
      if (existing) {
        existing.score += rrf;
      } else {
        scores.set(item.id, { item, score: rrf });
      }
    }
  }

  return Array.from(scores.values()).sort((a, b) => b.score - a.score);
}

// ─── Tool definition ──────────────────────────────────────────────────────────

export const searchKnowledgeTool: ToolDefinition<SearchKnowledgeInput, KnowledgeResult[]> = {
  name: "search_hytale_knowledge",
  description:
    "Federated semantic search across ALL Hytale knowledge bases at once: " +
    "server code, client UI, game data (items/NPCs/recipes/zones), and modding docs. " +
    "Use this when you're unsure which collection has the answer, or when the topic " +
    "spans multiple areas (e.g. 'how does crafting work' touches both code AND game data). " +
    "Results are fused with Reciprocal Rank Fusion for the best overall relevance.",
  inputSchema: searchKnowledgeSchema,

  async handler(input, context): Promise<ToolResult<KnowledgeResult[]>> {
    if (context.configError || !context.embedding) {
      return {
        success: false,
        error: context.configError || "Embedding provider not configured",
      };
    }

    const limit = Math.min(Math.max(1, input.limit ?? 8), 20);
    const sources = input.sources ?? ["code", "client", "gamedata", "docs"];

    // Embed once, reuse across all collections
    // Use "text" for the unified query — docs + gamedata dominate, code ANN still works
    const queryVector = await context.embedding.embedQuery(input.query, "text");
    const codeVector = sources.includes("code")
      ? await context.embedding.embedQuery(input.query, "code")
      : null;

    // Fetch candidates from each requested source (more than limit so RRF has material)
    const fetchN = Math.min(limit * 3, 20);

    const [codeRaw, clientRaw, gamedataRaw, docsRaw] = await Promise.all([
      sources.includes("code") && codeVector
        ? context.vectorStore
            .search(context.config.tables.code, codeVector, { limit: fetchN })
            .catch(() => [])
        : Promise.resolve([]),
      sources.includes("client")
        ? context.vectorStore
            .search(context.config.tables.clientUI, queryVector, { limit: fetchN })
            .catch(() => [])
        : Promise.resolve([]),
      sources.includes("gamedata")
        ? context.vectorStore
            .search(context.config.tables.gamedata, queryVector, { limit: fetchN })
            .catch(() => [])
        : Promise.resolve([]),
      sources.includes("docs")
        ? context.vectorStore
            .search(context.config.tables.docs, queryVector, { limit: fetchN })
            .catch(() => [])
        : Promise.resolve([]),
    ]);

    // Normalise each list into KnowledgeResult so RRF can work across sources
    const normalize = (
      source: KnowledgeResult["source"],
      results: Array<{ id: string; data: Record<string, unknown>; score: number }>
    ): KnowledgeResult[] =>
      results.map((r) => {
        const d = r.data;
        switch (source) {
          case "code":
            return {
              source,
              id: `code:${d.id as string}`,
              title: `${d.className as string}.${d.methodName as string}`,
              snippet: (d.methodSignature as string) || "",
              path: resolveCodePath(d.filePath as string),
              score: r.score,
              raw: d,
            };
          case "client":
            return {
              source,
              id: `client:${d.id as string}`,
              title: (d.name as string) || (d.id as string),
              snippet: ((d.content as string) || "").slice(0, 200),
              path: (d.relativePath as string) || (d.filePath as string) || "",
              score: r.score,
              raw: d,
            };
          case "gamedata":
            return {
              source,
              id: `gamedata:${d.id as string}`,
              title: `${d.name as string} (${d.type as string})`,
              snippet: ((d.rawJson as string) || "").slice(0, 200),
              path: resolveGameDataPath(d.filePath as string),
              score: r.score,
              raw: d,
            };
          case "docs":
            return {
              source,
              id: `docs:${d.id as string}`,
              title: (d.title as string) || (d.id as string),
              snippet: ((d.content as string) || "").slice(0, 200),
              path: (d.relativePath as string) || "",
              score: r.score,
              raw: d,
            };
        }
      });

    const lists: KnowledgeResult[][] = [
      normalize("code", codeRaw as any),
      normalize("client", clientRaw as any),
      normalize("gamedata", gamedataRaw as any),
      normalize("docs", docsRaw as any),
    ];

    // RRF merge
    let fused = rrfMerge(lists).slice(0, limit * 2); // extra headroom for reranker

    // Optional Voyage reranking over the fused pool
    if (context.rerankApiKey && fused.length > 0) {
      try {
        const reranked = await rerankResults(
          context.rerankApiKey,
          input.query,
          fused.map((x) => x.item),
          (r) => `${r.title}\n\n${r.snippet}`
        );
        fused = reranked.slice(0, limit).map((item) => ({ item, score: item.score }));
      } catch {
        // Non-fatal
      }
    }

    const data = fused.slice(0, limit).map((x) => x.item);
    return { success: true, data };
  },
};

// ─── Formatter ────────────────────────────────────────────────────────────────

const SOURCE_LABEL: Record<KnowledgeResult["source"], string> = {
  code: "Server Code",
  client: "Client UI",
  gamedata: "Game Data",
  docs: "Documentation",
};

export function formatKnowledgeResults(results: KnowledgeResult[]): string {
  if (results.length === 0) {
    return "No results found. Try rephrasing or searching a specific collection directly.";
  }

  return results
    .map((r, i) => {
      const lines = [
        `## Result ${i + 1}: ${r.title}`,
        `**Source:** ${SOURCE_LABEL[r.source]}`,
        `**Path:** ${r.path}`,
        `**Relevance:** ${(r.score * 100).toFixed(1)}%`,
        "",
        "```",
        r.snippet || "(no preview)",
        "```",
      ];
      return lines.join("\n");
    })
    .join("\n\n---\n\n");
}
