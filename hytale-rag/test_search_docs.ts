
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

async function main() {
    const transport = new StdioClientTransport({
        command: "npx",
        args: ["tsx", "src/index.ts"],
        env: { ...process.env, HYTALE_RAG_MODE: "mcp" }
    });

    const client = new Client(
        { name: "test-client", version: "1.0.0" },
        { capabilities: {} }
    );

    await client.connect(transport);
    console.log("Connected to MCP server");

    console.log("\nCalling search_hytale_docs...");
    const docsResult = await client.request({
        method: "tools/call",
        params: { 
            name: "search_hytale_docs", 
            arguments: { query: "inventory layout and hotbar" } 
        }
    }, CallToolResultSchema);
    console.log("Docs Search Result snippet:", JSON.stringify(docsResult.content, null, 2).substring(0, 1000));

    await transport.close();
}

main().catch(error => {
    console.error("Error:", error);
    process.exit(1);
});
