import type { EmbeddingPurpose } from "../../providers/embedding/interface.js";
import type { VectorSearchResult } from "../../providers/vectorstore/interface.js";
import type { SearchDetailLevel } from "../schemas.js";
import type { ToolContext } from "./index.js";

interface CacheEntry<T> {
  value: T;
  expiresAt: number;
  lastAccessedAt: number;
}

interface RankedVectorResult<T> {
  result: VectorSearchResult<T>;
  rerankScore: number;
  matchReasons: string[];
}

const MAX_CACHE_ENTRIES = 200;
const STOP_WORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "how",
  "i",
  "in",
  "is",
  "it",
  "of",
  "on",
  "or",
  "show",
  "that",
  "the",
  "this",
  "to",
  "use",
  "what",
  "where",
  "with",
]);

const queryEmbeddingCache = new Map<string, CacheEntry<number[]>>();
const responseCache = new Map<string, CacheEntry<unknown>>();

function pruneExpiredEntries<T>(cache: Map<string, CacheEntry<T>>): void {
  const now = Date.now();
  for (const [key, entry] of cache.entries()) {
    if (entry.expiresAt <= now) {
      cache.delete(key);
    }
  }

  if (cache.size <= MAX_CACHE_ENTRIES) {
    return;
  }

  const oldestEntries = Array.from(cache.entries())
    .sort((left, right) => left[1].lastAccessedAt - right[1].lastAccessedAt)
    .slice(0, cache.size - MAX_CACHE_ENTRIES);

  for (const [key] of oldestEntries) {
    cache.delete(key);
  }
}

function stableStringify(value: unknown): string {
  if (value === null || value === undefined) {
    return String(value);
  }

  if (typeof value !== "object") {
    return JSON.stringify(value);
  }

  if (Array.isArray(value)) {
    return `[${value.map((item) => stableStringify(item)).join(",")}]`;
  }

  const objectValue = value as Record<string, unknown>;
  const keys = Object.keys(objectValue).sort();
  const pairs = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(objectValue[key])}`);
  return `{${pairs.join(",")}}`;
}

function getCacheValue<T>(cache: Map<string, CacheEntry<T>>, key: string): T | undefined {
  pruneExpiredEntries(cache);
  const entry = cache.get(key);
  if (!entry) {
    return undefined;
  }

  if (entry.expiresAt <= Date.now()) {
    cache.delete(key);
    return undefined;
  }

  entry.lastAccessedAt = Date.now();
  return entry.value;
}

function setCacheValue<T>(cache: Map<string, CacheEntry<T>>, key: string, value: T, ttlMs: number): T {
  if (ttlMs <= 0) {
    return value;
  }

  cache.set(key, {
    value,
    expiresAt: Date.now() + ttlMs,
    lastAccessedAt: Date.now(),
  });
  pruneExpiredEntries(cache);
  return value;
}

function normalizeWhitespace(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function normalizeText(value: string): string {
  return normalizeWhitespace(value.toLowerCase());
}

export function getEffectiveDetail(
  detail: SearchDetailLevel | undefined,
  context: ToolContext
): SearchDetailLevel {
  if (detail) {
    return detail;
  }

  return context.config.retrieval.compactResponses ? "compact" : "balanced";
}

export async function getCachedQueryEmbedding(
  context: ToolContext,
  query: string,
  purpose: EmbeddingPurpose
): Promise<number[]> {
  if (!context.embedding) {
    throw new Error("Embedding provider not configured");
  }

  const ttlMs = context.config.retrieval.queryEmbeddingCacheTtlMs;
  const key = [
    context.config.embedding.provider,
    purpose,
    normalizeWhitespace(query).toLowerCase(),
  ].join("::");

  const cached = ttlMs > 0 ? getCacheValue(queryEmbeddingCache, key) : undefined;
  if (cached) {
    return cached;
  }

  const embedded = await context.embedding.embedQuery(query, purpose);
  return setCacheValue(queryEmbeddingCache, key, embedded, ttlMs);
}

export function getCachedToolResponse<T>(
  toolName: string,
  input: unknown,
  context: ToolContext
): T | undefined {
  const ttlMs = context.config.retrieval.responseCacheTtlMs;
  if (ttlMs <= 0) {
    return undefined;
  }

  const key = `${toolName}::${stableStringify(input)}`;
  return getCacheValue(responseCache, key) as T | undefined;
}

export function setCachedToolResponse<T>(
  toolName: string,
  input: unknown,
  value: T,
  context: ToolContext
): T {
  const ttlMs = context.config.retrieval.responseCacheTtlMs;
  const key = `${toolName}::${stableStringify(input)}`;
  return setCacheValue(responseCache as Map<string, CacheEntry<T>>, key, value, ttlMs);
}

export function extractQueryTerms(query: string): string[] {
  const rawTerms = query
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .split(/[^a-z0-9_./-]+/)
    .map((term) => term.trim())
    .filter((term) => term.length >= 3 && !STOP_WORDS.has(term));

  return Array.from(new Set(rawTerms));
}

function findRelevantWindow(content: string, terms: string[], maxChars: number): string {
  if (content.length <= maxChars || terms.length === 0) {
    return content;
  }

  const lowerContent = content.toLowerCase();
  let bestIndex = -1;

  for (const term of terms) {
    const index = lowerContent.indexOf(term);
    if (index !== -1 && (bestIndex === -1 || index < bestIndex)) {
      bestIndex = index;
    }
  }

  if (bestIndex === -1) {
    return content.slice(0, maxChars);
  }

  const halfWindow = Math.floor(maxChars / 2);
  let start = Math.max(0, bestIndex - halfWindow);
  let end = Math.min(content.length, start + maxChars);

  const nextLineBreak = content.lastIndexOf("\n", start);
  if (nextLineBreak !== -1) {
    start = nextLineBreak + 1;
    end = Math.min(content.length, start + maxChars);
  }

  return content.slice(start, end);
}

function getExcerptBudget(detail: SearchDetailLevel, contentType: "code" | "markup" | "json" | "text"): number {
  if (detail === "full") {
    return Number.MAX_SAFE_INTEGER;
  }

  if (detail === "balanced") {
    switch (contentType) {
      case "code":
        return 900;
      case "markup":
        return 1000;
      case "json":
        return 900;
      default:
        return 1000;
    }
  }

  switch (contentType) {
    case "code":
      return 420;
    case "markup":
      return 500;
    case "json":
      return 460;
    default:
      return 520;
  }
}

export function compactContent(
  content: string,
  query: string,
  detail: SearchDetailLevel,
  contentType: "code" | "markup" | "json" | "text"
): { value: string; truncated: boolean } {
  const budget = getExcerptBudget(detail, contentType);
  if (budget === Number.MAX_SAFE_INTEGER || content.length <= budget) {
    return { value: content, truncated: false };
  }

  const excerpt = findRelevantWindow(content, extractQueryTerms(query), budget).trimEnd();
  const prefix = excerpt !== content.slice(0, excerpt.length) ? "...\n" : "";
  const suffix = excerpt.length < content.length ? "\n..." : "";

  return {
    value: `${prefix}${excerpt}${suffix}`,
    truncated: true,
  };
}

export function rerankAndLimit<T>(
  results: VectorSearchResult<T>[],
  query: string,
  options: {
    limit: number;
    getSearchText: (data: T) => string;
    getDeduplicationKey: (data: T) => string;
  }
): RankedVectorResult<T>[] {
  const normalizedQuery = normalizeText(query);
  const terms = extractQueryTerms(query);

  const scored = results.map((result) => {
    const searchText = normalizeText(options.getSearchText(result.data));
    const matchedTerms = terms.filter((term) => searchText.includes(term));
    const hasExactPhrase = normalizedQuery.length >= 4 && searchText.includes(normalizedQuery);
    const lexicalBoost = Math.min(
      0.28,
      matchedTerms.length * 0.035 + (hasExactPhrase ? 0.12 : 0)
    );

    const matchReasons: string[] = [];
    if (hasExactPhrase) {
      matchReasons.push("exact phrase match");
    }
    if (matchedTerms.length > 0) {
      matchReasons.push(`keyword overlap: ${matchedTerms.slice(0, 4).join(", ")}`);
    }
    if (result.score >= 0.75) {
      matchReasons.push("high semantic similarity");
    }

    return {
      result,
      rerankScore: result.score + lexicalBoost,
      matchReasons,
    };
  });

  scored.sort((left, right) => {
    if (right.rerankScore !== left.rerankScore) {
      return right.rerankScore - left.rerankScore;
    }
    return right.result.score - left.result.score;
  });

  const selected: RankedVectorResult<T>[] = [];
  const seen = new Set<string>();

  for (const candidate of scored) {
    const dedupeKey = options.getDeduplicationKey(candidate.result.data);
    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    selected.push(candidate);

    if (selected.length >= options.limit) {
      break;
    }
  }

  return selected;
}

export function getFriendlyMissingTableMessage(tableName: string, error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/not found/i.test(message) || /dataset at path/i.test(message)) {
    return `Table '${tableName}' is not indexed locally. Run the matching ingest task or use hytale_index_health to confirm which datasets are available.`;
  }

  return message;
}