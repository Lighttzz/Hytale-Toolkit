/**
 * Documentation Parser
 *
 * Parses MDX/Markdown documentation files from the HytaleModding.dev site.
 * These files contain modding guides, tutorials, and reference documentation.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

/**
 * Types of documentation content
 */
export type DocsType =
  | "guide"         // Tutorial/guide content
  | "reference"     // API reference documentation
  | "faq"           // Frequently asked questions
  | "example";      // Code examples

/**
 * A chunk of documentation content ready for embedding
 */
export interface DocsChunk {
  id: string;                    // e.g., "guide:plugin/creating-block"
  type: DocsType;
  title: string;                 // Extracted from frontmatter or filename
  filePath: string;              // Full path to the file
  relativePath: string;          // Path relative to content/docs/en
  fileHash: string;              // SHA-256 hash for incremental indexing
  content: string;               // Raw file content (MDX/`Markdown`)
  category?: string;             // e.g., "guides", "plugin", "ecs"
  description?: string;          // From frontmatter if available
  textForEmbedding: string;      // Text optimized for semantic search
}

/**
 * Result from parsing documentation files
 */
export interface DocsParseResult {
  chunks: DocsChunk[];
  errors: string[];
}

/**
 * Statistics about the documentation
 */
export interface DocsStats {
  totalFiles: number;
  byCategory: Record<string, number>;
  byType: Record<string, number>;
}

/**
 * Extract frontmatter from MDX/Markdown content
 */
function extractFrontmatter(content: string): { frontmatter: Record<string, string>; body: string } {
  const frontmatterMatch = content.match(/^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/);

  if (!frontmatterMatch) {
    return { frontmatter: {}, body: content };
  }

  const frontmatterStr = frontmatterMatch[1];
  const body = frontmatterMatch[2];
  const frontmatter: Record<string, string> = {};

  // Parse simple YAML-like frontmatter
  for (const line of frontmatterStr.split("\n")) {
    const match = line.match(/^(\w+):\s*(.+)$/);
    if (match) {
      let value = match[2].trim();
      // Remove quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      frontmatter[match[1]] = value;
    }
  }

  return { frontmatter, body };
}

/**
 * Determine the documentation type from file path and content
 */
function getDocsType(relativePath: string, content: string): DocsType {
  const pathLower = relativePath.toLowerCase();

  if (pathLower.includes("faq")) return "faq";
  if (pathLower.includes("example") || pathLower.includes("sample")) return "example";
  if (pathLower.includes("reference") || pathLower.includes("api")) return "reference";

  // Check content for code-heavy files
  const codeBlocks = (content.match(/```/g) || []).length;
  if (codeBlocks >= 6) return "example";

  return "guide";
}

/**
 * Extract category from file path
 */
function extractCategory(relativePath: string): string | undefined {
  const parts = relativePath.split(/[/\\]/);

  // Return the first meaningful directory
  if (parts.length > 1) {
    return parts[0];
  }

  return undefined;
}

/**
 * Extract title from frontmatter or filename
 */
function extractTitle(frontmatter: Record<string, string>, filePath: string): string {
  // Try frontmatter fields
  if (frontmatter.title) return frontmatter.title;
  if (frontmatter.name) return frontmatter.name;

  // Fall back to filename
  const basename = path.basename(filePath, path.extname(filePath));
  // Convert kebab-case to Title Case
  return basename
    .split("-")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Strip MDX/JSX components and imports for cleaner embedding text
 */
function stripMdxComponents(content: string): string {
  let cleaned = content;

  // Remove import statements
  cleaned = cleaned.replace(/^import\s+.*?from\s+['"].*?['"];?\s*$/gm, "");
  cleaned = cleaned.replace(/^import\s+['"].*?['"];?\s*$/gm, "");

  // Remove JSX component tags (but keep the content inside)
  // e.g., <Callout type="info">content</Callout> -> content
  cleaned = cleaned.replace(/<(\w+)[^>]*>([\s\S]*?)<\/\1>/g, "$2");

  // Remove self-closing JSX tags
  cleaned = cleaned.replace(/<\w+[^>]*\/>/g, "");

  // Clean up extra whitespace
  cleaned = cleaned.replace(/\n{3,}/g, "\n\n");

  return cleaned.trim();
}

/**
 * Build embedding text for a documentation chunk
 */
function buildEmbeddingText(chunk: DocsChunk, body: string): string {
  const cleanedBody = stripMdxComponents(body);

  const parts = [
    `# ${chunk.title}`,
    "",
    `Category: ${chunk.category || "General"}`,
    `Type: ${chunk.type}`,
  ];

  if (chunk.description) {
    parts.push(`Description: ${chunk.description}`);
  }

  parts.push("");
  parts.push("This is documentation for Hytale modding from HytaleModding.dev.");
  parts.push("");
  parts.push(cleanedBody);

  return parts.join("\n");
}

/**
 * Parse a single documentation file
 */
function parseDocsFile(
  filePath: string,
  basePath: string
): DocsChunk | null {
  const ext = path.extname(filePath).toLowerCase();
  if (ext !== ".mdx" && ext !== ".md") return null;

  let content: string;
  try {
    content = fs.readFileSync(filePath, "utf-8");
  } catch {
    return null;
  }

  // Skip empty files
  if (content.trim().length === 0) return null;

  // Skip meta.json files that might have wrong extension
  if (content.trim().startsWith("{")) return null;

  const { frontmatter, body } = extractFrontmatter(content);
  const relativePath = path.relative(basePath, filePath).replace(/\\/g, "/");
  const title = extractTitle(frontmatter, filePath);
  const category = extractCategory(relativePath);
  const type = getDocsType(relativePath, content);
  const fileHash = crypto.createHash("sha256").update(content).digest("hex");

  // Create ID from relative path without extension
  const idPath = relativePath.replace(/\.(mdx?|md)$/, "");

  const chunk: DocsChunk = {
    id: `${type}:${idPath}`,
    type,
    title,
    filePath,
    relativePath,
    fileHash,
    content,
    category,
    description: frontmatter.description,
    textForEmbedding: "", // Will be set below
  };

  chunk.textForEmbedding = buildEmbeddingText(chunk, body);

  return chunk;
}

/**
 * Recursively collect all documentation files from a directory
 */
function collectDocsFiles(dir: string): string[] {
  const files: string[] = [];

  function walk(currentDir: string) {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(currentDir, { withFileTypes: true });
    } catch {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(currentDir, entry.name);

      if (entry.isDirectory()) {
        walk(fullPath);
      } else if (entry.isFile()) {
        const ext = path.extname(entry.name).toLowerCase();
        if (ext === ".mdx" || ext === ".md") {
          files.push(fullPath);
        }
      }
    }
  }

  walk(dir);
  return files;
}

/**
 * Parse all documentation files from a directory
 */
export async function parseDocsDirectory(
  docsPath: string,
  onProgress?: (current: number, total: number, file: string) => void
): Promise<DocsParseResult> {
  const chunks: DocsChunk[] = [];
  const errors: string[] = [];

  // Collect all docs files
  const files = collectDocsFiles(docsPath);

  // Parse each file
  for (let i = 0; i < files.length; i++) {
    const file = files[i];

    if (onProgress) {
      onProgress(i + 1, files.length, file);
    }

    try {
      const chunk = parseDocsFile(file, docsPath);
      if (chunk) {
        chunks.push(chunk);
      }
    } catch (e: any) {
      errors.push(`Error parsing ${file}: ${e.message}`);
    }
  }

  return { chunks, errors };
}

/**
 * Get statistics about the documentation files
 */
export async function getDocsStats(docsPath: string): Promise<DocsStats> {
  const files = collectDocsFiles(docsPath);

  const stats: DocsStats = {
    totalFiles: 0,
    byCategory: {},
    byType: {},
  };

  for (const file of files) {
    const relativePath = path.relative(docsPath, file);
    const category = extractCategory(relativePath) || "Other";

    // Read file to determine type
    let content = "";
    try {
      content = fs.readFileSync(file, "utf-8");
    } catch {
      continue;
    }

    // Skip non-doc files
    if (content.trim().startsWith("{")) continue;

    const type = getDocsType(relativePath, content);

    stats.totalFiles++;
    stats.byCategory[category] = (stats.byCategory[category] || 0) + 1;
    stats.byType[type] = (stats.byType[type] || 0) + 1;
  }

  return stats;
}
