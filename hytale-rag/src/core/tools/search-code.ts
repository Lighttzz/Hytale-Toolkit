/**
 * Search Code Tool
 *
 * Semantic search over the decompiled Hytale codebase.
 */

import { searchCodeSchema, type SearchCodeInput } from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import type { CodeSearchResult } from "../types.js";
import { resolveCodePath } from "../../utils/paths.js";
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
 * Search code tool definition
 */
export const searchCodeTool: ToolDefinition<SearchCodeInput, CodeSearchResult[]> = {
  name: "search_hytale_code",
  description:
    "Search the decompiled Hytale codebase using semantic search. " +
    "Use this to find methods, classes, or functionality by describing what you're looking for. " +
    "Returns relevant Java methods with compact excerpts by default. " +
    "Set detail='full' when you need the full method body.",
  inputSchema: searchCodeSchema,

  async handler(input, context): Promise<ToolResult<CodeSearchResult[]>> {
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
    const cached = getCachedToolResponse<CodeSearchResult[]>(searchCodeTool.name, cacheInput, context);
    if (cached) {
      return { success: true, data: cached };
    }

    const domainConfig = context.config.retrieval.domains.code;
    const candidateLimit = Math.max(limit, domainConfig.candidatePoolSize);

    const queryVector = await getCachedQueryEmbedding(context, input.query, "code");

    // Build filter
    const filter = input.classFilter
      ? { className: input.classFilter }
      : undefined;

    // Search
    let results;
    try {
      results = await context.vectorStore.search<CodeSearchResult>(
        context.config.tables.code,
        queryVector,
        { limit: candidateLimit, filter, minScore: domainConfig.minScore }
      );

      if (results.length === 0 && domainConfig.minScore > 0) {
        results = await context.vectorStore.search<CodeSearchResult>(
          context.config.tables.code,
          queryVector,
          { limit: candidateLimit, filter }
        );
      }
    } catch (error) {
      return {
        success: false,
        error: getFriendlyMissingTableMessage(context.config.tables.code, error),
      };
    }

    const reranked = rerankAndLimit(results, input.query, {
      limit,
      getSearchText: (data) => [
        data.className,
        data.methodName,
        data.methodSignature,
        data.packageName,
        data.filePath,
        data.content.slice(0, 2200),
      ].join("\n"),
      getDeduplicationKey: (data) => `${data.className}:${data.methodName}:${data.filePath}`,
    });

    const data: CodeSearchResult[] = reranked.map(({ result, matchReasons }) => {
      const compacted = compactContent(result.data.content, input.query, detail, "code");
      return {
        id: result.data.id,
        className: result.data.className,
        packageName: result.data.packageName,
        methodName: result.data.methodName,
        methodSignature: result.data.methodSignature,
        content: compacted.value,
        filePath: result.data.filePath,
        lineStart: result.data.lineStart,
        lineEnd: result.data.lineEnd,
        score: result.score,
        matchReasons,
        truncated: compacted.truncated,
        detail,
      };
    });

    return { success: true, data: setCachedToolResponse(searchCodeTool.name, cacheInput, data, context) };
  },
};

/**
 * Format code search results as markdown (for MCP/display)
 */
export function formatCodeResults(results: CodeSearchResult[]): string {
  if (results.length === 0) {
    return "No results found for your query.";
  }

  return results
    .map((r, i) => {
      const fullPath = resolveCodePath(r.filePath);
      return `## Result ${i + 1}: ${r.className}.${r.methodName}
**Package:** ${r.packageName}
**File:** ${fullPath}:${r.lineStart}-${r.lineEnd}
**Signature:** \`${r.methodSignature}\`
**Relevance:** ${(r.score * 100).toFixed(1)}%
    ${r.matchReasons && r.matchReasons.length > 0 ? `**Why it matched:** ${r.matchReasons.join("; ")}\n` : ""}${r.truncated ? "**Payload:** excerpted for token efficiency\n" : ""}

\`\`\`java
${r.content}
\`\`\``;
    })
    .join("\n\n---\n\n");
}
