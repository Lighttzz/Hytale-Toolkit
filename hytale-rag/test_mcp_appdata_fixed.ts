import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { ListToolsResultSchema, CallToolResultSchema } from "@modelcontextprotocol/sdk/types.js";

async function main() {
    const transport = new StdioClientTransport({
        command: "powershell",
        args: ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\\Users\\xpvpx\\AppData\\Local\\Hytale-Toolkit\\hytale-rag\\start-mcp.ps1"],
        env: { ...process.env, HYTALE_RAG_MODE: "mcp" }
    });

    const client = new Client(
        { name: "test-client-appdata", version: "1.0.0" },
        { capabilities: {} }
    );

    await client.connect(transport);
    console.log("Connected to AppData MCP server");

    const query = process.env.SEARCH_QUERY || "how do I create a hytale command";

    const toolsResult = await client.request({ method: "tools/list" }, ListToolsResultSchema);
    const tools = toolsResult.tools;
    console.log("Available tools:", tools.map(t => t.name).join(", "));

    // Since hytale_index_health and search_hytale_knowledge are missing, let's use search_hytale_docs
    if (tools.some(t => t.name === "search_hytale_docs")) {
        console.log("\nCalling search_hytale_docs...");
        const docsResult = await client.request({
            method: "tools/call",
            params: { name: "search_hytale_docs", arguments: { query } }
        }, CallToolResultSchema);
        console.log("Docs Search Result:\n", JSON.stringify(docsResult.content, null, 2));
    }

    if (tools.some(t => t.name === "search_hytale_code")) {
        console.log("\nCalling search_hytale_code...");
        const codeResult = await client.request({
            method: "tools/call",
            params: { name: "search_hytale_code", arguments: { query } }
        }, CallToolResultSchema);
        console.log("Code Search Result:\n", JSON.stringify(codeResult.content, null, 2));
    }

    await transport.close();
}

main().catch(error => {
    console.error("Error:", error);
    process.exit(1);
});
