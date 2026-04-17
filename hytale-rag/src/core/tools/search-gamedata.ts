/**
 * Search Game Data Tool
 *
 * Semantic search over Hytale game data (items, recipes, NPCs, etc.)
 */

import { searchGameDataSchema, type SearchGameDataInput } from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import type { GameDataSearchResult, GameDataType } from "../types.js";
import { resolveGameDataPath } from "../../utils/paths.js";
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
 * Search game data tool definition
 */
export const searchGameDataTool: ToolDefinition<SearchGameDataInput, GameDataSearchResult[]> = {
  name: "search_hytale_gamedata",
  description:
    "Search vanilla Hytale game data including items, recipes, NPCs, drops, blocks, and more. " +
    "Use this for modding questions like 'how to craft X', 'what drops Y', 'NPC behavior for Z', " +
    "'what items use tag T', or 'how does the farming system work'. " +
    "Returns compact excerpts by default.",
  inputSchema: searchGameDataSchema,

  async handler(input, context): Promise<ToolResult<GameDataSearchResult[]>> {
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
    const cached = getCachedToolResponse<GameDataSearchResult[]>(searchGameDataTool.name, cacheInput, context);
    if (cached) {
      return { success: true, data: cached };
    }

    const domainConfig = context.config.retrieval.domains.gamedata;
    const candidateLimit = Math.max(limit, domainConfig.candidatePoolSize);

    const queryVector = await getCachedQueryEmbedding(context, input.query, "text");

    // Build filter (null/undefined means no filter)
    const typeFilter =
      input.type && input.type !== "all"
        ? { type: input.type as GameDataType }
        : undefined;

    // Search
    let results;
    try {
      results = await context.vectorStore.search<GameDataSearchResult>(
        context.config.tables.gamedata,
        queryVector,
        { limit: candidateLimit, filter: typeFilter, minScore: domainConfig.minScore }
      );

      if (results.length === 0 && domainConfig.minScore > 0) {
        results = await context.vectorStore.search<GameDataSearchResult>(
          context.config.tables.gamedata,
          queryVector,
          { limit: candidateLimit, filter: typeFilter }
        );
      }
    } catch (error) {
      return {
        success: false,
        error: getFriendlyMissingTableMessage(context.config.tables.gamedata, error),
      };
    }

    const reranked = rerankAndLimit(results, input.query, {
      limit,
      getSearchText: (data) => [
        data.name,
        data.type,
        data.category,
        data.filePath,
        data.tags?.join(" "),
        data.parentId,
        data.rawJson.slice(0, 2200),
      ].join("\n"),
      getDeduplicationKey: (data) => data.filePath,
    });

    const data: GameDataSearchResult[] = reranked.map(({ result, matchReasons }) => {
      const compacted = compactContent(result.data.rawJson, input.query, detail, "json");
      return {
        id: result.data.id,
        type: result.data.type,
        name: result.data.name,
        filePath: result.data.filePath,
        rawJson: compacted.value,
        category: result.data.category,
        tags: result.data.tags || [],
        parentId: result.data.parentId,
        score: result.score,
        matchReasons,
        truncated: compacted.truncated,
        detail,
      };
    });

    return { success: true, data: setCachedToolResponse(searchGameDataTool.name, cacheInput, data, context) };
  },
};

/**
 * Format game data search results as markdown (for MCP/display)
 */
export function formatGameDataResults(results: GameDataSearchResult[]): string {
  if (results.length === 0) {
    return "No game data found for your query.";
  }

  return results
    .map((r, i) => {
      const fullPath = resolveGameDataPath(r.filePath);
      const parts = [
        `## Result ${i + 1}: ${r.name}`,
        `**Type:** ${r.type}`,
        `**Path:** ${fullPath}`,
      ];

      if (r.category) parts.push(`**Category:** ${r.category}`);
      if (r.parentId) parts.push(`**Parent:** ${r.parentId}`);
      if (r.tags && r.tags.length > 0) parts.push(`**Tags:** ${r.tags.join(", ")}`);
      parts.push(`**Relevance:** ${(r.score * 100).toFixed(1)}%`);
      if (r.matchReasons && r.matchReasons.length > 0) {
        parts.push(`**Why it matched:** ${r.matchReasons.join("; ")}`);
      }
      if (r.truncated) {
        parts.push("**Payload:** excerpted for token efficiency");
      }

      // Pretty print the JSON
      let jsonContent = r.rawJson;
      try {
        jsonContent = JSON.stringify(JSON.parse(r.rawJson), null, 2);
      } catch {
        // Keep original if parse fails
      }

      parts.push("");
      parts.push("```json");
      parts.push(jsonContent);
      parts.push("```");

      return parts.join("\n");
    })
    .join("\n\n---\n\n");
}
