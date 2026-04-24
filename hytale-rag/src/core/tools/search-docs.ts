/**
 * Search Documentation Tool
 *
 * Semantic search over HytaleModding.dev documentation.
 */

import { searchDocsSchema, type SearchDocsInput } from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import type { DocsSearchResult } from "../types.js";

/**
 * Search documentation tool definition
 */
export const searchDocsTool: ToolDefinition<SearchDocsInput, DocsSearchResult[]> = {
  name: "search_hytale_docs",
  description:
    "Search HytaleModding.dev documentation using semantic search. " +
    "Use this to find modding guides, tutorials, and reference documentation. " +
    "Covers topics like plugin development, ECS, block creation, commands, events, and more.",
  inputSchema: searchDocsSchema,

  async handler(input, context): Promise<ToolResult<DocsSearchResult[]>> {
    // Check for configuration errors (e.g., missing API key)
    if (context.configError || !context.embedding) {
      return {
        success: false,
        error: context.configError || "Embedding provider not configured",
      };
    }

    // Clamp limit
    const limit = Math.min(Math.max(1, input.limit ?? 5), 20);

    // Get embedding for the query (use "text" since docs are prose, not code)
    const queryVector = await context.embedding.embedQuery(input.query, "text");

    // Build filter for type if not "all"
    const filter = input.type && input.type !== "all"
      ? { type: input.type }
      : undefined;

    // Search
    const results = await context.vectorStore.search<DocsSearchResult>(
      context.config.tables.docs,
      queryVector,
      { limit, filter }
    );

    // Map results
    const data: DocsSearchResult[] = results.map((r) => ({
      id: r.data.id,
      type: r.data.type,
      title: r.data.title,
      filePath: r.data.filePath,
      relativePath: r.data.relativePath,
      content: r.data.content,
      category: r.data.category,
      description: r.data.description,
      score: r.score,
    }));

    return { success: true, data };
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

      // Truncate content if too long (keep first ~2000 chars)
      let content = r.content;
      if (content.length > 2000) {
        content = content.substring(0, 2000) + "\n\n... (content truncated)";
      }

      return `${header}
${metadata.join("\n")}

\`\`\`markdown
${content}
\`\`\``;
    })
    .join("\n\n---\n\n");
}
