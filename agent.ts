/**
 * @fileoverview HR Chatbot Agent using LangChain, MongoDB, and Anthropic's Claude
 * This system implements a stateful conversational agent that can search and retrieve
 * employee information using vector similarity search.
 */

import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { StateGraph } from "@langchain/langgraph";
import { Annotation } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { z } from "zod";
import "dotenv/config";

/**
 * Main function to handle agent interactions with the HR database
 * @param {MongoClient} client - MongoDB client instance
 * @param {string} query - User's query text
 * @param {string} thread_id - Unique identifier for the conversation thread
 * @returns {Promise<string>} The agent's response to the query
 */
export async function callAgent(client: MongoClient, query: string, thread_id: string) {
  // Initialize database connection
  const dbName = "hr_database";
  const db = client.db(dbName);
  const collection = db.collection("employees");

  /**
   * Define the graph state structure using LangGraph annotations
   * This maintains the conversation history and state between interactions
   */
  const GraphState = Annotation.Root({
    messages: Annotation<BaseMessage[]>({
      reducer: (x, y) => x.concat(y), // Combines message arrays
    }),
  });

  /**
   * Tool definition for employee lookup functionality
   * This tool performs vector similarity search in MongoDB
   */
  const employeeLookupTool = tool(
    async ({ query, n = 10 }) => {
      console.log("Employee lookup tool called");

      // Configure MongoDB Atlas Vector Search
      const dbConfig = {
        collection: collection,
        indexName: "vector_index",
        textKey: "embedding_text",
        embeddingKey: "embedding",
      };

      // Initialize vector store with embeddings
      const vectorStore = new MongoDBAtlasVectorSearch(
        new OpenAIEmbeddings(),
        dbConfig
      );

      // Perform similarity search and return results
      const result = await vectorStore.similaritySearchWithScore(query, n);
      return JSON.stringify(result);
    },
    {
      name: "employee_lookup",
      description: "Gathers employee details from the HR database",
      schema: z.object({
        query: z.string().describe("The search query"),
        n: z
          .number()
          .optional()
          .default(10)
          .describe("Number of results to return"),
      }),
    }
  );

  // Array of available tools for the agent
  const tools = [employeeLookupTool];

  // Initialize tool node with state typing
  const toolNode = new ToolNode<typeof GraphState.State>(tools);

  /**
   * Initialize the Anthropic Claude model with tool bindings
   * Using temperature 0 for more consistent, factual responses
   */
  const model = new ChatAnthropic({
    model: "claude-3-5-sonnet-20240620",
    temperature: 0,
  }).bindTools(tools);

  /**
   * Determines the next step in the conversation flow
   * @param {typeof GraphState.State} state - Current state of the conversation
   * @returns {string} Next node to execute ("tools" or "__end__")
   */
  function shouldContinue(state: typeof GraphState.State) {
    const messages = state.messages;
    const lastMessage = messages[messages.length - 1] as AIMessage;

    // Route to tools if the model wants to use them
    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    // End conversation if no tools needed
    return "__end__";
  }

  /**
   * Handles model interaction using the current conversation state
   * @param {typeof GraphState.State} state - Current state of the conversation
   * @returns {Promise<{messages: AIMessage[]}>} New messages to add to the state
   */
  async function callModel(state: typeof GraphState.State) {
    // Define the conversation template
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful AI assistant, collaborating with other assistants. Use the provided tools to progress towards answering the question. If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. Execute what you can to make progress. If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop. You have access to the following tools: {tool_names}.\n{system_message}\nCurrent time: {time}.`,
      ],
      new MessagesPlaceholder("messages"),
    ]);

    // Format the prompt with current context
    const formattedPrompt = await prompt.formatMessages({
      system_message: "You are helpful HR Chatbot Agent.",
      time: new Date().toISOString(),
      tool_names: tools.map((tool) => tool.name).join(", "),
      messages: state.messages,
    });

    // Get model response
    const result = await model.invoke(formattedPrompt);

    return { messages: [result] };
  }

  /**
   * Define the conversation workflow graph
   * This creates a state machine that manages the flow between the agent and tools
   */
  const workflow = new StateGraph(GraphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue)
    .addEdge("tools", "agent");

  // Initialize MongoDB state persistence
  const checkpointer = new MongoDBSaver({ client, dbName });

  // Compile the workflow into a runnable application
  const app = workflow.compile({ checkpointer });

  /**
   * Execute the conversation workflow
   * @param {HumanMessage} messages - Initial user message
   * @param {Object} options - Configuration options including thread ID
   * @returns {Promise<typeof GraphState.State>} Final conversation state
   */
  const finalState = await app.invoke(
    {
      messages: [new HumanMessage(query)],
    },
    { recursionLimit: 15, configurable: { thread_id: thread_id } }
  );

  // Return the last message content from the conversation
  console.log(finalState.messages[finalState.messages.length - 1].content);
  return finalState.messages[finalState.messages.length - 1].content;
}