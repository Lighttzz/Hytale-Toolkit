import {
  searchKnowledgeSchema,
  type SearchKnowledgeInput,
  type SearchKnowledgeDomain,
} from "../schemas.js";
import type { ToolDefinition, ToolContext, ToolResult } from "./index.js";
import { searchClientCodeTool, type ClientUISearchResult } from "./search-client-code.js";
import { searchCodeTool } from "./search-code.js";
import { searchDocsTool } from "./search-docs.js";
import { searchGameDataTool } from "./search-gamedata.js";
import type { CodeSearchResult, DocsSearchResult, GameDataSearchResult } from "../types.js";
import { getCachedToolResponse, getEffectiveDetail, extractQueryTerms, setCachedToolResponse } from "./retrieval.js";

type SearchableDomain = Exclude<SearchKnowledgeDomain, "auto">;

export interface KnowledgeSearchItem {
  domain: SearchableDomain;
  title: string;
  location: string;
  score: number;
  excerpt: string;
  matchReasons?: string[];
}

export interface KnowledgeSearchResponse {
  results: KnowledgeSearchItem[];
  searchedDomains: SearchableDomain[];
  unavailableDomains: SearchableDomain[];
}

const DOMAIN_TABLES: Record<SearchableDomain, keyof ToolContext["config"]["tables"]> = {
  code: "code",
  client_ui: "clientUI",
  gamedata: "gamedata",
  docs: "docs",
};

const DOMAIN_HINTS: Array<{ domain: SearchableDomain; keywords: string[] }> = [
  { domain: "client_ui", keywords: ["ui", "xaml", "hud", "inventory", "hotbar", "menu", "overlay", "screen", ".ui"] },
  { domain: "gamedata", keywords: ["recipe", "craft", "drop", "npc", "item", "biome", "block", "zone", "prefab", "weather", "seed"] },
  { domain: "code", keywords: ["class", "method", "function", "listener", "dispatch", "implementation", "java", "source", "code"] },
  { domain: "docs", keywords: ["guide", "tutorial", "docs", "documentation", "how", "plugin", "modding", "reference"] },
];

function rankDomains(query: string): SearchableDomain[] {
  const lowerQuery = query.toLowerCase();
  const terms = extractQueryTerms(query);
  const scores = new Map<SearchableDomain, number>([
    ["code", 0],
    ["client_ui", 0],
    ["gamedata", 0],
    ["docs", 0],
  ]);

  for (const { domain, keywords } of DOMAIN_HINTS) {
    let score = scores.get(domain) ?? 0;
    for (const keyword of keywords) {
      if (lowerQuery.includes(keyword) || terms.includes(keyword)) {
        score += keyword.length <= 4 ? 2 : 3;
      }
    }
    scores.set(domain, score);
  }

  if (lowerQuery.includes("how to") || lowerQuery.includes("where do")) {
    scores.set("docs", (scores.get("docs") ?? 0) + 2);
    scores.set("gamedata", (scores.get("gamedata") ?? 0) + 1);
  }

  return Array.from(scores.entries())
    .sort((left, right) => right[1] - left[1])
    .map(([domain]) => domain);
}

function toKnowledgeItemFromCode(result: CodeSearchResult): KnowledgeSearchItem {
  return {
    domain: "code",
    title: `${result.className}.${result.methodName}`,
    location: `${result.filePath}:${result.lineStart}-${result.lineEnd}`,
    score: result.score,
    excerpt: result.content,
    matchReasons: result.matchReasons,
  };
}

function toKnowledgeItemFromClientUI(result: ClientUISearchResult): KnowledgeSearchItem {
  return {
    domain: "client_ui",
    title: result.name,
    location: result.relativePath,
    score: result.score,
    excerpt: result.content,
    matchReasons: result.matchReasons,
  };
}

function toKnowledgeItemFromGameData(result: GameDataSearchResult): KnowledgeSearchItem {
  return {
    domain: "gamedata",
    title: `${result.type}: ${result.name}`,
    location: result.filePath,
    score: result.score,
    excerpt: result.rawJson,
    matchReasons: result.matchReasons,
  };
}

function toKnowledgeItemFromDocs(result: DocsSearchResult): KnowledgeSearchItem {
  return {
    domain: "docs",
    title: result.title,
    location: result.relativePath,
    score: result.score,
    excerpt: result.content,
    matchReasons: result.matchReasons,
  };
}

function mergeAndRankResults(
  orderedDomains: SearchableDomain[],
  resultSets: Partial<Record<SearchableDomain, KnowledgeSearchItem[]>>,
  limit: number
): KnowledgeSearchItem[] {
  const domainBoost = new Map<SearchableDomain, number>(orderedDomains.map((domain, index) => [domain, Math.max(0, 0.08 - index * 0.03)]));
  const merged = orderedDomains.flatMap((domain) => (resultSets[domain] ?? []).map((item, index) => ({
    item,
    blendedScore: item.score + (domainBoost.get(domain) ?? 0) - index * 0.01,
  })));

  merged.sort((left, right) => right.blendedScore - left.blendedScore);
  return merged.slice(0, limit).map((entry) => entry.item);
}

export const searchKnowledgeTool: ToolDefinition<SearchKnowledgeInput, KnowledgeSearchResponse> = {
  name: "search_hytale_knowledge",
  description:
    "Routed, token-efficient search across Hytale code, client UI, docs, and game data. " +
    "Use this first when you are unsure which index has the answer or want to avoid multiple tool calls.",
  inputSchema: searchKnowledgeSchema,

  async handler(input, context): Promise<ToolResult<KnowledgeSearchResponse>> {
    if (context.configError || !context.embedding) {
      return {
        success: false,
        error: context.configError || "Embedding provider not configured",
      };
    }

    const detail = getEffectiveDetail(input.detail, context);
    const limit = Math.min(Math.max(1, input.limit ?? 5), 10);
    const cacheInput = { ...input, detail, limit };
    const cached = getCachedToolResponse<KnowledgeSearchResponse>(searchKnowledgeTool.name, cacheInput, context);
    if (cached) {
      return { success: true, data: cached };
    }

    const preferredDomains = input.domain === "auto"
      ? rankDomains(input.query)
      : [input.domain];

    const orderedDomains: SearchableDomain[] = [];
    const unavailableDomains: SearchableDomain[] = [];
    for (const domain of preferredDomains) {
      const tableName = context.config.tables[DOMAIN_TABLES[domain]];
      const exists = await context.vectorStore.tableExists(tableName);
      if (!exists) {
        unavailableDomains.push(domain);
        continue;
      }
      orderedDomains.push(domain);
    }

    if (orderedDomains.length === 0) {
      return {
        success: false,
        error: "No indexed domains are available for this search. Use hytale_index_health to confirm which datasets are present locally.",
      };
    }

    const domainsToSearch = input.domain === "auto"
      ? orderedDomains.slice(0, orderedDomains.length > 1 ? 2 : 1)
      : orderedDomains.slice(0, 1);

    const resultSets: Partial<Record<SearchableDomain, KnowledgeSearchItem[]>> = {};

    for (const domain of domainsToSearch) {
      if (domain === "code") {
        const response = await searchCodeTool.handler({ query: input.query, limit: Math.min(limit, 4), detail }, context);
        if (response.success && response.data) {
          resultSets.code = response.data.map(toKnowledgeItemFromCode);
        }
        continue;
      }

      if (domain === "client_ui") {
        const response = await searchClientCodeTool.handler({ query: input.query, limit: Math.min(limit, 4), detail }, context);
        if (response.success && response.data) {
          resultSets.client_ui = response.data.map(toKnowledgeItemFromClientUI);
        }
        continue;
      }

      if (domain === "gamedata") {
        const response = await searchGameDataTool.handler({ query: input.query, type: "all", limit: Math.min(limit, 4), detail }, context);
        if (response.success && response.data) {
          resultSets.gamedata = response.data.map(toKnowledgeItemFromGameData);
        }
        continue;
      }

      const response = await searchDocsTool.handler({ query: input.query, type: "all", limit: Math.min(limit, 4), detail }, context);
      if (response.success && response.data) {
        resultSets.docs = response.data.map(toKnowledgeItemFromDocs);
      }
    }

    const responseData: KnowledgeSearchResponse = {
      results: mergeAndRankResults(domainsToSearch, resultSets, limit),
      searchedDomains: domainsToSearch,
      unavailableDomains,
    };

    return {
      success: true,
      data: setCachedToolResponse(searchKnowledgeTool.name, cacheInput, responseData, context),
    };
  },
};

export function formatKnowledgeResults(response: KnowledgeSearchResponse): string {
  if (response.results.length === 0) {
    const unavailable = response.unavailableDomains.length > 0
      ? ` Missing local indices: ${response.unavailableDomains.join(", ")}.`
      : "";
    return `No results found for your query.${unavailable}`;
  }

  const lines: string[] = [
    "# Hytale Knowledge Results",
    "",
    `**Searched domains:** ${response.searchedDomains.join(", ")}`,
  ];

  if (response.unavailableDomains.length > 0) {
    lines.push(`**Unavailable locally:** ${response.unavailableDomains.join(", ")}`);
  }

  lines.push("");

  for (const [index, result] of response.results.entries()) {
    lines.push(`## Result ${index + 1}: ${result.title}`);
    lines.push(`**Domain:** ${result.domain}`);
    lines.push(`**Location:** ${result.location}`);
    lines.push(`**Relevance:** ${(result.score * 100).toFixed(1)}%`);
    if (result.matchReasons && result.matchReasons.length > 0) {
      lines.push(`**Why it matched:** ${result.matchReasons.join("; ")}`);
    }
    lines.push("");
    lines.push("```");
    lines.push(result.excerpt);
    lines.push("```");
    lines.push("");
  }

  return lines.join("\n").trimEnd();
}