/**
 * Search Client UI Tool
 *
 * Semantic search over Hytale client UI files (.xaml, .ui, .json).
 */

import { searchClientCodeSchema, type SearchClientCodeInput } from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import { resolveClientDataPath } from "../../utils/paths.js";
import {
  compactContent,
  getCachedQueryEmbedding,
  getCachedToolResponse,
  getEffectiveDetail,
  getFriendlyMissingTableMessage,
  rerankAndLimit,
  setCachedToolResponse,
} from "./retrieval.js";

/**
 * Client UI search result
 */
export interface ClientUISearchResult {
  id: string;
  type: string;
  name: string;
  filePath: string;
  relativePath: string;
  content: string;
  category?: string;
  score: number;
  matchReasons?: string[];
  truncated?: boolean;
  detail?: "compact" | "balanced" | "full";
}

/**
 * Search client UI tool definition
 */
export const searchClientCodeTool: ToolDefinition<SearchClientCodeInput, ClientUISearchResult[]> = {
  name: "search_hytale_client_code",
  description:
    "Search Hytale client UI files using semantic search. " +
    "Use this to find UI templates (.xaml), UI components (.ui), and NodeEditor definitions. " +
    "Useful for modifying game UI appearance like inventory layout, hotbar, health bars, etc. " +
    "Returns compact excerpts by default.",
  inputSchema: searchClientCodeSchema,

  async handler(input, context): Promise<ToolResult<ClientUISearchResult[]>> {
    // Check for configuration errors (e.g., missing API key)
    if (context.configError || !context.embedding) {
      return {
        success: false,
        error: context.configError || "Embedding provider not configured",
      };
    }

    const detail = getEffectiveDetail(input.detail, context);
    const limit = Math.min(Math.max(1, input.limit ?? 5), 20);
    const cacheInput = { ...input, detail, limit };
    const cached = getCachedToolResponse<ClientUISearchResult[]>(searchClientCodeTool.name, cacheInput, context);
    if (cached) {
      return { success: true, data: cached };
    }

    const domainConfig = context.config.retrieval.domains.clientUI;
    const candidateLimit = Math.max(limit, domainConfig.candidatePoolSize);

    const queryVector = await getCachedQueryEmbedding(context, input.query, "text");

    // Build filter
    const filter = input.classFilter
      ? { category: input.classFilter }
      : undefined;

    // Search
    let results;
    try {
      results = await context.vectorStore.search<ClientUISearchResult>(
        context.config.tables.clientUI,
        queryVector,
        { limit: candidateLimit, filter, minScore: domainConfig.minScore }
      );

      if (results.length === 0 && domainConfig.minScore > 0) {
        results = await context.vectorStore.search<ClientUISearchResult>(
          context.config.tables.clientUI,
          queryVector,
          { limit: candidateLimit, filter }
        );
      }
    } catch (error) {
      return {
        success: false,
        error: getFriendlyMissingTableMessage(context.config.tables.clientUI, error),
      };
    }

    const reranked = rerankAndLimit(results, input.query, {
      limit,
      getSearchText: (data) => [
        data.name,
        data.relativePath,
        data.category,
        data.type,
        data.content.slice(0, 2200),
      ].join("\n"),
      getDeduplicationKey: (data) => data.relativePath,
    });

    const data: ClientUISearchResult[] = reranked.map(({ result, matchReasons }) => {
      const compacted = compactContent(result.data.content, input.query, detail, "markup");
      return {
        id: result.data.id,
        type: result.data.type,
        name: result.data.name,
        filePath: result.data.filePath,
        relativePath: result.data.relativePath,
        content: compacted.value,
        category: result.data.category,
        score: result.score,
        matchReasons,
        truncated: compacted.truncated,
        detail,
      };
    });

    return { success: true, data: setCachedToolResponse(searchClientCodeTool.name, cacheInput, data, context) };
  },
};

/**
 * Format client UI search results as markdown (for MCP/display)
 */
export function formatClientUIResults(results: ClientUISearchResult[]): string {
  if (results.length === 0) {
    return "No results found for your query in the client UI files.";
  }

  return results
    .map((r, i) => {
      const fullPath = resolveClientDataPath(r.filePath);
      const fileType = r.type === "xaml" ? "xml" : r.type === "ui" ? "css" : "json";
      return `## Result ${i + 1}: ${r.name}
**Type:** ${r.type.toUpperCase()}
**Category:** ${r.category || "General"}
**Path:** ${fullPath}
**Relevance:** ${(r.score * 100).toFixed(1)}%
    ${r.matchReasons && r.matchReasons.length > 0 ? `**Why it matched:** ${r.matchReasons.join("; ")}\n` : ""}${r.truncated ? "**Payload:** excerpted for token efficiency\n" : ""}

\`\`\`${fileType}
${r.content}
\`\`\``;
    })
    .join("\n\n---\n\n");
}
