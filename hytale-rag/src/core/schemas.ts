/**
 * Tool Input Schemas
 *
 * Zod schemas for tool input validation.
 * These schemas are shared across all server implementations.
 */

import { z } from "zod";

/**
 * Valid game data types for filtering
 */
export const GAME_DATA_TYPES = [
  "all",
  "item",
  "recipe",
  "block",
  "interaction",
  "drop",
  "npc",
  "npc_group",
  "npc_ai",
  "entity",
  "projectile",
  "farming",
  "shop",
  "environment",
  "weather",
  "biome",
  "worldgen",
  "camera",
  "objective",
  "gameplay",
  "localization",
  "zone",
  "terrain_layer",
  "cave",
  "prefab",
] as const;

export type GameDataTypeFilter = (typeof GAME_DATA_TYPES)[number];

/**
 * Code search input schema
 */
export const searchCodeSchema = z.object({
  query: z
    .string()
    .min(1)
    .describe("Natural language description of what you're looking for"),
  limit: z
    .number()
    .int()
    .min(1)
    .max(20)
    .optional()
    .default(5)
    .describe("Number of results to return (default 5, max 20)"),
  classFilter: z
    .string()
    .optional()
    .describe("Filter results to a specific class name"),
});

export type SearchCodeInput = z.infer<typeof searchCodeSchema>;

/**
 * Client UI search input schema
 */
export const searchClientCodeSchema = z.object({
  query: z
    .string()
    .min(1)
    .describe("Natural language description of what UI element you're looking for"),
  limit: z
    .number()
    .int()
    .min(1)
    .max(20)
    .optional()
    .default(5)
    .describe("Number of results to return (default 5, max 20)"),
  classFilter: z
    .string()
    .optional()
    .describe("Filter results to a specific category (e.g., DesignSystem, InGame, MainMenu)"),
});

export type SearchClientCodeInput = z.infer<typeof searchClientCodeSchema>;

/**
 * Game data search input schema
 */
export const searchGameDataSchema = z.object({
  query: z
    .string()
    .min(1)
    .describe("Natural language question about Hytale game data"),
  type: z
    .enum(GAME_DATA_TYPES)
    .optional()
    .default("all")
    .describe("Filter by data type (default: all)"),
  limit: z
    .number()
    .int()
    .min(1)
    .max(20)
    .optional()
    .default(5)
    .describe("Number of results (default 5, max 20)"),
});

export type SearchGameDataInput = z.infer<typeof searchGameDataSchema>;

/**
 * Valid documentation types for filtering
 */
export const DOCS_TYPES = [
  "all",
  "guide",
  "reference",
  "faq",
  "example",
] as const;

export type DocsTypeFilter = (typeof DOCS_TYPES)[number];

/**
 * Documentation search input schema
 */
export const searchDocsSchema = z.object({
  query: z
    .string()
    .min(1)
    .describe("Natural language question about Hytale modding"),
  type: z
    .enum(DOCS_TYPES)
    .optional()
    .default("all")
    .describe("Filter by documentation type (default: all)"),
  limit: z
    .number()
    .int()
    .min(1)
    .max(20)
    .optional()
    .default(5)
    .describe("Number of results (default 5, max 20)"),
});

export type SearchDocsInput = z.infer<typeof searchDocsSchema>;

/**
 * Empty schema for tools with no parameters
 */
export const emptySchema = z.object({});

export type EmptyInput = z.infer<typeof emptySchema>;
