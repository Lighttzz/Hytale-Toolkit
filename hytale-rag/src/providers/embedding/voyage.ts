/**
 * Voyage AI Embedding Provider
 *
 * Implementation of EmbeddingProvider for Voyage AI.
 * Uses voyage-code-3 for code and voyage-4-large for text/gamedata.
 *
 * Supports HTTPS_PROXY/HTTP_PROXY environment variables for users
 * in regions where Voyage AI may be geo-blocked.
 */

import {
  EmbeddingProvider,
  EmbeddingProviderConfig,
  EmbeddingOptions,
  EmbeddingResult,
  EmbeddingPurpose,
  EmbeddingProgressCallback,
} from "./interface.js";
import { ProxyAgent, type Dispatcher } from "undici";

const VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings";

/**
 * Get proxy dispatcher if HTTPS_PROXY or HTTP_PROXY is configured.
 * Returns undefined if no proxy is set.
 */
function getProxyDispatcher(): Dispatcher | undefined {
  const proxyUrl = process.env.HTTPS_PROXY || process.env.HTTP_PROXY ||
                   process.env.https_proxy || process.env.http_proxy;

  if (!proxyUrl) {
    return undefined;
  }

  return new ProxyAgent(proxyUrl);
}

/** Default models for different purposes */
const DEFAULT_MODELS = {
  code: "voyage-code-3",
  text: "voyage-4-large",
};

/** Vector dimensions by model (all support 256, 512, 1024, 2048 via Matryoshka) */
const MODEL_DIMENSIONS: Record<string, number> = {
  "voyage-code-3": 1024,
  "voyage-4-large": 1024,
};

/** Default batch size (Voyage supports up to 128) */
const DEFAULT_BATCH_SIZE = 128;

/** Max elapsed wait between retries (ms) */
const MAX_RETRY_WAIT_MS = 64_000;
/** Max number of retry attempts on 429 */
const MAX_RETRIES = 6;

/** Max characters before truncation (~8K tokens for code) */
const MAX_CHARS_CODE = 32000;
const MAX_CHARS_TEXT = 16000;

/**
 * Voyage AI response format
 */
interface VoyageResponse {
  data: Array<{ embedding: number[]; index: number }>;
  usage: { total_tokens: number };
}

/**
 * Truncate text to token limit
 */
function truncateToLimit(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  return text.substring(0, maxChars) + "\n// ... truncated";
}

/**
 * Voyage AI Embedding Provider
 */
export class VoyageEmbeddingProvider implements EmbeddingProvider {
  readonly name = "voyage";

  private apiKey: string;
  private models: { code: string; text: string };
  private batchSize: number;
  private dispatcher: Dispatcher | undefined;

  constructor(config: EmbeddingProviderConfig) {
    if (!config.apiKey) {
      throw new Error("Voyage API key is required. Set VOYAGE_API_KEY environment variable.");
    }

    this.apiKey = config.apiKey;
    this.models = {
      code: config.models?.code || DEFAULT_MODELS.code,
      text: config.models?.text || DEFAULT_MODELS.text,
    };
    this.batchSize = config.batchSize || DEFAULT_BATCH_SIZE;
    this.dispatcher = getProxyDispatcher();
  }

  /**
   * Execute a Voyage API fetch with exponential backoff on 429 responses.
   *
   * Voyage tier-1 limits: 2000 RPM / 3M TPM for voyage-4-large and voyage-code-3.
   * Strategy: send at full speed; when a 429 arrives honour the Retry-After header if
   * present, otherwise back off exponentially (2^attempt seconds ± 20 % jitter, capped
   * at MAX_RETRY_WAIT_MS) and retry the same batch.
   */
  private async fetchWithBackoff(
    body: string,
    attempt: number = 0
  ): Promise<Response> {
    const response = await fetch(VOYAGE_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body,
      // @ts-expect-error - dispatcher is a valid undici option for Node.js fetch
      dispatcher: this.dispatcher,
    });

    if (response.status !== 429) {
      return response;
    }

    if (attempt >= MAX_RETRIES) {
      const errorText = await response.text();
      throw new Error(`Voyage API rate limited after ${MAX_RETRIES} retries: ${errorText}`);
    }

    // Prefer Retry-After header; fall back to exponential backoff with ±20% jitter.
    const retryAfterHeader = response.headers.get("retry-after");
    let waitMs: number;
    if (retryAfterHeader) {
      const seconds = parseFloat(retryAfterHeader);
      waitMs = isNaN(seconds) ? 1000 : Math.ceil(seconds * 1000);
    } else {
      const base = Math.pow(2, attempt) * 1000; // 1 s, 2 s, 4 s, …
      waitMs = Math.min(base * (0.8 + Math.random() * 0.4), MAX_RETRY_WAIT_MS);
    }

    process.stderr.write(
      `\n  [Voyage] Rate limited (429) — waiting ${(waitMs / 1000).toFixed(1)}s before retry ${attempt + 1}/${MAX_RETRIES}...\n`
    );
    await new Promise<void>((resolve) => setTimeout(resolve, waitMs));

    return this.fetchWithBackoff(body, attempt + 1);
  }

  /**
   * Embed multiple texts in batches
   */
  async embedBatch(
    texts: string[],
    options: EmbeddingOptions,
    onProgress?: EmbeddingProgressCallback
  ): Promise<EmbeddingResult> {
    const model = options.purpose === "code" ? this.models.code : this.models.text;
    const inputType = options.mode === "query" ? "query" : "document";
    const maxChars = options.purpose === "code" ? MAX_CHARS_CODE : MAX_CHARS_TEXT;

    const allVectors: number[][] = [];
    let totalTokens = 0;

    // Truncate texts
    const truncatedTexts = texts.map((t) => truncateToLimit(t, maxChars));

    // Process in batches
    for (let i = 0; i < truncatedTexts.length; i += this.batchSize) {
      const batch = truncatedTexts.slice(i, i + this.batchSize);

      const response = await this.fetchWithBackoff(
        JSON.stringify({ model, input: batch, input_type: inputType })
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Voyage API error: ${response.status} - ${errorText}`);
      }

      const data = (await response.json()) as VoyageResponse;

      // Sort by index to maintain order
      data.data.sort((a, b) => a.index - b.index);
      allVectors.push(...data.data.map((d) => d.embedding));
      totalTokens += data.usage.total_tokens;

      // Report progress
      if (onProgress) {
        onProgress(Math.min(i + this.batchSize, truncatedTexts.length), truncatedTexts.length);
      }
    }

    return {
      vectors: allVectors,
      model,
      dimensions: this.getDimensions(options.purpose),
      usage: { totalTokens },
    };
  }

  /**
   * Embed a single query
   */
  async embedQuery(text: string, purpose: EmbeddingPurpose): Promise<number[]> {
    const result = await this.embedBatch([text], { purpose, mode: "query" });
    return result.vectors[0];
  }

  /**
   * Get vector dimensions for a purpose
   */
  getDimensions(purpose: EmbeddingPurpose): number {
    const model = purpose === "code" ? this.models.code : this.models.text;
    return MODEL_DIMENSIONS[model] || 1024;
  }

  /**
   * Validate the provider is configured correctly
   */
  async validate(): Promise<boolean> {
    try {
      const response = await this.fetchWithBackoff(
        JSON.stringify({ model: this.models.text, input: ["test"], input_type: "query" })
      );
      return response.ok;
    } catch {
      return false;
    }
  }
}
