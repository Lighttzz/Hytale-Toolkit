/**
 * Search Documentation Tool
 *
 * Semantic search over HytaleModding.dev documentation.
 */

import { searchDocsSchema, type SearchDocsInput } from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import type { DocsSearchResult } from "../types.js";
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
 * Search documentation tool definition
 */
export const searchDocsTool: ToolDefinition<SearchDocsInput, DocsSearchResult[]> = {
  name: "search_hytale_docs",
  description:
    "Search HytaleModding.dev documentation using semantic search. " +
    "Use this to find modding guides, tutorials, and reference documentation. " +
    "Covers topics like plugin development, ECS, block creation, commands, events, and more. " +
    "Returns compact excerpts by default.",
  inputSchema: searchDocsSchema,

  async handler(input, context): Promise<ToolResult<DocsSearchResult[]>> {
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
    const cached = getCachedToolResponse<DocsSearchResult[]>(searchDocsTool.name, cacheInput, context);
    if (cached) {
      return { success: true, data: cached };
    }

    const domainConfig = context.config.retrieval.domains.docs;
    const candidateLimit = Math.max(limit, domainConfig.candidatePoolSize);

    const queryVector = await getCachedQueryEmbedding(context, input.query, "text");

    // Build filter for type if not "all"
    const filter = input.type && input.type !== "all"
      ? { type: input.type }
      : undefined;

    // Search
    let results;
    try {
      results = await context.vectorStore.search<DocsSearchResult>(
        context.config.tables.docs,
        queryVector,
        { limit: candidateLimit, filter, minScore: domainConfig.minScore }
      );

      if (results.length === 0 && domainConfig.minScore > 0) {
        results = await context.vectorStore.search<DocsSearchResult>(
          context.config.tables.docs,
          queryVector,
          { limit: candidateLimit, filter }
        );
      }
    } catch (error) {
      return {
        success: false,
        error: getFriendlyMissingTableMessage(context.config.tables.docs, error),
      };
    }

    const reranked = rerankAndLimit(results, input.query, {
      limit,
      getSearchText: (data) => [
        data.title,
        data.relativePath,
        data.category,
        data.description,
        data.content.slice(0, 2400),
      ].join("\n"),
      getDeduplicationKey: (data) => data.relativePath,
    });

    const data: DocsSearchResult[] = reranked.map(({ result, matchReasons }) => {
      const compacted = compactContent(result.data.content, input.query, detail, "text");
      return {
        id: result.data.id,
        type: result.data.type,
        title: result.data.title,
        filePath: result.data.filePath,
        relativePath: result.data.relativePath,
        content: compacted.value,
        category: result.data.category,
        description: result.data.description,
        score: result.score,
        matchReasons,
        truncated: compacted.truncated,
        detail,
      };
    });

    return { success: true, data: setCachedToolResponse(searchDocsTool.name, cacheInput, data, context) };
  },
};

/**
 * Format documentation search results as markdown (for MCP/display)
 */
export function formatDocsResults(results: DocsSearchResult[]): string {
  if (results.length === 0) {
    return "No documentation found for your query. Try a different search term or check if the docs have been indexed.";
  }

  return results
    .map((r, i) => {
      const header = `## Result ${i + 1}: ${r.title}`;
      const metadata = [
        `**Type:** ${r.type}`,
        `**Category:** ${r.category || "General"}`,
        `**Path:** ${r.relativePath}`,
        `**Relevance:** ${(r.score * 100).toFixed(1)}%`,
      ];

      if (r.description) {
        metadata.push(`**Description:** ${r.description}`);
      }

      if (r.matchReasons && r.matchReasons.length > 0) {
        metadata.push(`**Why it matched:** ${r.matchReasons.join("; ")}`);
      }
      if (r.truncated) {
        metadata.push("**Payload:** excerpted for token efficiency");
      }

      return `${header}
${metadata.join("\n")}

\`\`\`markdown
${r.content}
\`\`\``;
    })
    .join("\n\n---\n\n");
}
