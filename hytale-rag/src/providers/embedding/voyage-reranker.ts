/**
 * Voyage AI Reranker
 *
 * Uses Voyage's rerank API (rerank-2.5 / rerank-2.5-lite) to re-score a candidate
 * set returned by the initial vector search.  Reranking is optional and only
 * fires when VOYAGE_API_KEY is available.
 *
 * Supports HTTPS_PROXY / HTTP_PROXY environment variables.
 */

import { ProxyAgent, type Dispatcher } from "undici";

const VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank";

/**
 * Default rerank model.
 * rerank-2.5       – highest quality, slower
 * rerank-2.5-lite  – faster, cheaper, still much better than pure ANN
 */
export const DEFAULT_RERANK_MODEL = "rerank-2.5";

/** Max docs that can be re-ranked per request (Voyage limit is 1000). */
const VOYAGE_RERANK_MAX_DOCS = 50;

export interface RerankItem {
  index: number;
  relevanceScore: number;
}

interface VoyageRerankResponse {
  data: RerankItem[];
  usage: { total_tokens: number };
}

function getProxyDispatcher(): Dispatcher | undefined {
  const url =
    process.env.HTTPS_PROXY ||
    process.env.HTTP_PROXY ||
    process.env.https_proxy ||
    process.env.http_proxy;
  return url ? new ProxyAgent(url) : undefined;
}

/**
 * Rerank a set of candidate documents against a query.
 *
 * @param apiKey     Voyage API key
 * @param query      The user's search query
 * @param documents  Candidate document texts (max VOYAGE_RERANK_MAX_DOCS)
 * @param model      Rerank model name
 * @returns          Array of { index, relevanceScore } sorted by descending relevance
 */
export async function rerankDocuments(
  apiKey: string,
  query: string,
  documents: string[],
  model: string = DEFAULT_RERANK_MODEL
): Promise<RerankItem[]> {
  const docs = documents.slice(0, VOYAGE_RERANK_MAX_DOCS);

  const body = JSON.stringify({
    model,
    query,
    documents: docs,
    return_documents: false,
  });

  const dispatcher = getProxyDispatcher();

  const response = await fetch(VOYAGE_RERANK_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body,
    // @ts-expect-error – undici dispatcher option for Node.js fetch
    dispatcher,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Voyage rerank error: ${response.status} – ${text}`);
  }

  const data = (await response.json()) as VoyageRerankResponse;
  // Already sorted by descending relevance by the API
  return data.data;
}

/**
 * Re-rank an array of typed results using their text content.
 *
 * @param apiKey       Voyage API key
 * @param query        Search query string
 * @param results      Original vector-search results (any type)
 * @param getText      Function to extract the text snippet from a result
 * @param model        Rerank model (default: rerank-2-lite)
 * @returns            Results re-ordered by rerank relevance score, with score replaced
 */
export async function rerankResults<T extends { score: number }>(
  apiKey: string,
  query: string,
  results: T[],
  getText: (result: T) => string,
  model: string = DEFAULT_RERANK_MODEL
): Promise<T[]> {
  if (results.length === 0) return results;

  const texts = results.map(getText);
  const ranked = await rerankDocuments(apiKey, query, texts, model);

  // Build reordered list with updated scores
  return ranked.map((item) => ({
    ...results[item.index],
    score: item.relevanceScore,
  }));
}
