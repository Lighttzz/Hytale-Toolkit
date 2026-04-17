
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { ListToolsResultSchema, CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

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

    const query = process.env.SEARCH_QUERY || "inventory layout and hotbar";

    // 2. List tools
    const toolsResult = await client.request({ method: "tools/list" }, ListToolsResultSchema);
    const tools = toolsResult.tools;
    const toolNames = tools.map(t => t.name);
    console.log("Available tools:", toolNames.join(", "));

    const hasSearchHytaleKnowledge = toolNames.includes("search_hytale_knowledge");
    console.log("search_hytale_knowledge exists:", hasSearchHytaleKnowledge);

    // 3. Verify descriptions
    const searchHytaleKnowledge = tools.find(t => t.name === "search_hytale_knowledge");
    const searchHytaleClientCode = tools.find(t => t.name === "search_hytale_client_code");

    if (searchHytaleKnowledge) {
        console.log("search_hytale_knowledge description:", searchHytaleKnowledge.description);
    }
    if (searchHytaleClientCode) {
        console.log("search_hytale_client_code description:", searchHytaleClientCode.description);
    }

    // 4. Call hytale_index_health
    console.log("\nCalling hytale_index_health...");
    const healthResult = await client.request({
        method: "tools/call",
        params: { name: "hytale_index_health", arguments: {} }
    }, CallToolResultSchema);
    console.log("Health Result:", JSON.stringify(healthResult.content, null, 2));

    // 5. Call search_hytale_knowledge
    console.log("\nCalling search_hytale_knowledge...");
    const knowledgeResult = await client.request({
        method: "tools/call",
        params: { 
            name: "search_hytale_knowledge", 
            arguments: { query: query } 
        }
    }, CallToolResultSchema);
    console.log("Knowledge Search Result:", JSON.stringify(knowledgeResult.content, null, 2));

    // 6. Call search_hytale_client_code
    if (searchHytaleClientCode) {
        console.log("\nCalling search_hytale_client_code...");
        const codeResult = await client.request({
            method: "tools/call",
            params: { 
                name: "search_hytale_client_code", 
                arguments: { query: "inventory layout and hotbar" } 
            }
        }, CallToolResultSchema);
        console.log("Code Search Result snippet:", JSON.stringify(codeResult.content, null, 2).substring(0, 1000));
    }

    await transport.close();
}

main().catch(error => {
    console.error("Error:", error);
    process.exit(1);
});
