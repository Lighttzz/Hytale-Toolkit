/**
 * Shared types for Hytale game data indexing.
 */

// Document types based on file path within Assets.zip
export type GameDataType =
  | "item"           // Server/Item/Items/**
  | "recipe"         // Server/Item/Recipes/**
  | "block"          // Server/Item/Block/**
  | "interaction"    // Server/Item/Interactions/** + RootInteractions/**
  | "drop"           // Server/Drops/**
  | "npc"            // Server/NPC/Roles/**
  | "npc_group"      // Server/NPC/Groups/**
  | "npc_ai"         // Server/NPC/DecisionMaking/**
  | "entity"         // Server/Entity/**
  | "projectile"     // Server/Projectiles/**
  | "farming"        // Server/Farming/**
  | "shop"           // Server/BarterShops/**
  | "environment"    // Server/Environments/**
  | "weather"        // Server/Weathers/**
  | "biome"          // Server/HytaleGenerator/Biomes/**
  | "worldgen"       // Server/HytaleGenerator/Assignments/**
  | "camera"         // Server/Camera/**
  | "objective"      // Server/Objective/**
  | "gameplay"       // Server/GameplayConfigs/**
  | "localization"   // Common/Languages/**
  | "zone"           // Server/World/**/Zones/**/*.json (tiles, customs, zones)
  | "terrain_layer"  // Server/World/**/Zones/**/Layers/**
  | "cave"           // Server/World/**/Zones/**/Cave/**
  | "prefab";        // Server/Prefabs/**

// A chunk of game data ready for embedding
export interface GameDataChunk {
  id: string;                    // e.g., "item:Soil_Clay_Smooth_Blue"
  type: GameDataType;
  name: string;                  // Human-readable name or item ID
  filePath: string;              // Path within Assets.zip
  fileHash: string;              // SHA-256 hash for incremental indexing

  // Raw content
  rawJson: string;               // Original JSON content (for display)

  // Searchable metadata
  category?: string;
  tags?: string[];
  parentId?: string;             // For inheritance (Parent field)
  relatedIds?: string[];         // Referenced items, NPCs, etc.

  // For embedding
  textForEmbedding: string;
}

// A chunk with its embedding vector
export interface EmbeddedGameDataChunk extends GameDataChunk {
  vector: number[];
}

// Search result returned from the database
export interface GameDataSearchResult {
  id: string;
  type: GameDataType;
  name: string;
  filePath: string;
  rawJson: string;
  category?: string;
  tags: string[];
  parentId?: string;
  score: number;
}

// Stats about indexed game data
export interface GameDataStats {
  totalItems: number;
  byType: Record<GameDataType, number>;
}

// ============ Documentation Types ============

// Types of documentation content
export type DocsType =
  | "guide"         // Tutorial/guide content
  | "reference"     // API reference documentation
  | "faq"           // Frequently asked questions
  | "example";      // Code examples

// A chunk of documentation ready for embedding
export interface DocsChunk {
  id: string;                    // e.g., "guide:plugin/creating-block"
  type: DocsType;
  title: string;                 // Extracted from frontmatter or filename
  filePath: string;              // Full path to the file
  relativePath: string;          // Path relative to content/docs/en
  fileHash: string;              // SHA-256 hash for incremental indexing
  content: string;               // Raw file content (MDX/Markdown)
  category?: string;             // e.g., "guides", "plugin", "ecs"
  description?: string;          // From frontmatter if available
  textForEmbedding: string;      // Text optimized for semantic search
}

// A docs chunk with its embedding vector
export interface EmbeddedDocsChunk extends DocsChunk {
  vector: number[];
}

// Search result returned from the docs database
export interface DocsSearchResult {
  id: string;
  type: DocsType;
  title: string;
  filePath: string;
  relativePath: string;
  content: string;
  category?: string;
  description?: string;
  score: number;
}

// Stats about indexed documentation
export interface DocsStats {
  totalDocs: number;
  byCategory: Record<string, number>;
  byType: Record<DocsType, number>;
}
