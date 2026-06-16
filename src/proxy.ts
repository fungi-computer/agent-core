/**
 * Proxy stream function for apps that route LLM calls through a server.
 * The server manages auth and proxies requests to LLM providers.
 */

// Internal import for JSON parsing utility
import {
  type AssistantMessage,
  type AssistantMessageEvent,
  type Context,
  EventStream,
  type Model,
  parseStreamingJson,
  type SimpleStreamOptions,
  type StopReason,
  type ToolCall,
} from "@mariozechner/pi-ai";

/**
 * Proxy event types - server sends these with partial field stripped to reduce bandwidth.
 */
export type ProxyAssistantMessageEvent =
  | { contentIndex: number; contentSignature?: string; type: "text_end" }
  | { contentIndex: number; contentSignature?: string; type: "thinking_end" }
  | { contentIndex: number; delta: string; type: "text_delta" }
  | { contentIndex: number; delta: string; type: "thinking_delta" }
  | { contentIndex: number; delta: string; type: "toolcall_delta" }
  | {
      contentIndex: number;
      id: string;
      toolName: string;
      type: "toolcall_start";
    }
  | { contentIndex: number; type: "text_start" }
  | { contentIndex: number; type: "thinking_start" }
  | { contentIndex: number; type: "toolcall_end" }
  | {
      errorMessage?: string;
      reason: Extract<StopReason, "aborted" | "error">;
      type: "error";
      usage: AssistantMessage["usage"];
    }
  | {
      reason: Extract<StopReason, "length" | "stop" | "toolUse">;
      type: "done";
      usage: AssistantMessage["usage"];
    }
  | { type: "start" };

export interface ProxyStreamOptions extends SimpleStreamOptions {
  /** Auth token for the proxy server */
  authToken: string;
  /** Proxy server URL (e.g., "https://genai.example.com") */
  proxyUrl: string;
}

// Create stream class matching ProxyMessageEventStream
class ProxyMessageEventStream extends EventStream<
  AssistantMessageEvent,
  AssistantMessage
> {
  constructor() {
    super(
      (event) => event.type === "done" || event.type === "error",
      (event) => {
        if (event.type === "done") return event.message;
        if (event.type === "error") return event.error;
        throw new Error("Unexpected event type");
      },
    );
  }
}

/**
 * Stream function that proxies through a server instead of calling LLM providers directly.
 * The server strips the partial field from delta events to reduce bandwidth.
 * We reconstruct the partial message client-side.
 *
 * Use this as the `streamFn` option when creating an Agent that needs to go through a proxy.
 *
 * @example
 * ```typescript
 * const agent = new Agent({
 *   streamFn: (model, context, options) =>
 *     streamProxy(model, context, {
 *       ...options,
 *       authToken: await getAuthToken(),
 *       proxyUrl: "https://genai.example.com",
 *     }),
 * });
 * ```
 */
export function streamProxy(
  model: Model<any>,
  context: Context,
  options: ProxyStreamOptions,
): ProxyMessageEventStream {
  const stream = new ProxyMessageEventStream();

  (async () => {
    // Initialize the partial message that we'll build up from events
    const partial: AssistantMessage = {
      api: model.api,
      content: [],
      model: model.id,
      provider: model.provider,
      role: "assistant",
      stopReason: "stop",
      timestamp: Date.now(),
      usage: {
        cacheRead: 0,
        cacheWrite: 0,
        cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
        input: 0,
        output: 0,
        totalTokens: 0,
      },
    };

    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;

    const abortHandler = () => {
      if (reader) {
        reader.cancel("Request aborted by user").catch(() => {});
      }
    };

    if (options.signal) {
      options.signal.addEventListener("abort", abortHandler);
    }

    try {
      const response = await fetch(`${options.proxyUrl}/api/stream`, {
        body: JSON.stringify({
          context,
          model,
          options: {
            maxTokens: options.maxTokens,
            reasoning: options.reasoning,
            temperature: options.temperature,
          },
        }),
        headers: {
          Authorization: `Bearer ${options.authToken}`,
          "Content-Type": "application/json",
        },
        method: "POST",
        signal: options.signal,
      });

      if (!response.ok) {
        let errorMessage = `Proxy error: ${response.status} ${response.statusText}`;
        try {
          const errorData = (await response.json()) as { error?: string };
          if (errorData.error) {
            errorMessage = `Proxy error: ${errorData.error}`;
          }
        } catch {
          // Couldn't parse error response
        }
        throw new Error(errorMessage);
      }

      reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        if (options.signal?.aborted) {
          throw new Error("Request aborted by user");
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data) {
              const proxyEvent = JSON.parse(data) as ProxyAssistantMessageEvent;
              const event = processProxyEvent(proxyEvent, partial);
              if (event) {
                stream.push(event);
              }
            }
          }
        }
      }

      if (options.signal?.aborted) {
        throw new Error("Request aborted by user");
      }

      stream.end();
    } catch (error) {
      console.error("[proxy] Stream error:", error);
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      const reason = options.signal?.aborted ? "aborted" : "error";
      partial.stopReason = reason;
      partial.errorMessage = errorMessage;
      stream.push({
        error: partial,
        reason,
        type: "error",
      });
      stream.end();
    } finally {
      if (options.signal) {
        options.signal.removeEventListener("abort", abortHandler);
      }
    }
  })();

  return stream;
}

/**
 * Process a proxy event and update the partial message.
 */
function processProxyEvent(
  proxyEvent: ProxyAssistantMessageEvent,
  partial: AssistantMessage,
): AssistantMessageEvent | undefined {
  switch (proxyEvent.type) {
    case "done":
      partial.stopReason = proxyEvent.reason;
      partial.usage = proxyEvent.usage;
      return { message: partial, reason: proxyEvent.reason, type: "done" };

    case "error":
      partial.stopReason = proxyEvent.reason;
      partial.errorMessage = proxyEvent.errorMessage;
      partial.usage = proxyEvent.usage;
      return { error: partial, reason: proxyEvent.reason, type: "error" };

    case "start":
      return { partial, type: "start" };

    case "text_delta": {
      const content = partial.content[proxyEvent.contentIndex];
      if (content?.type === "text") {
        content.text += proxyEvent.delta;
        return {
          contentIndex: proxyEvent.contentIndex,
          delta: proxyEvent.delta,
          partial,
          type: "text_delta",
        };
      }
      throw new Error("Received text_delta for non-text content");
    }

    case "text_end": {
      const content = partial.content[proxyEvent.contentIndex];
      if (content?.type === "text") {
        content.textSignature = proxyEvent.contentSignature;
        return {
          content: content.text,
          contentIndex: proxyEvent.contentIndex,
          partial,
          type: "text_end",
        };
      }
      throw new Error("Received text_end for non-text content");
    }

    case "thinking_start":
      partial.content[proxyEvent.contentIndex] = {
        thinking: "",
        type: "thinking",
      };
      return {
        contentIndex: proxyEvent.contentIndex,
        partial,
        type: "thinking_start",
      };

    case "text_start":
      partial.content[proxyEvent.contentIndex] = { text: "", type: "text" };
      return {
        contentIndex: proxyEvent.contentIndex,
        partial,
        type: "text_start",
      };

    case "thinking_delta": {
      const content = partial.content[proxyEvent.contentIndex];
      if (content?.type === "thinking") {
        content.thinking += proxyEvent.delta;
        return {
          contentIndex: proxyEvent.contentIndex,
          delta: proxyEvent.delta,
          partial,
          type: "thinking_delta",
        };
      }
      throw new Error("Received thinking_delta for non-thinking content");
    }

    case "thinking_end": {
      const content = partial.content[proxyEvent.contentIndex];
      if (content?.type === "thinking") {
        content.thinkingSignature = proxyEvent.contentSignature;
        return {
          content: content.thinking,
          contentIndex: proxyEvent.contentIndex,
          partial,
          type: "thinking_end",
        };
      }
      throw new Error("Received thinking_end for non-thinking content");
    }

    case "toolcall_start":
      partial.content[proxyEvent.contentIndex] = {
        arguments: {},
        id: proxyEvent.id,
        name: proxyEvent.toolName,
        partialJson: "",
        type: "toolCall",
      } satisfies { partialJson: string } & ToolCall as ToolCall;
      return {
        contentIndex: proxyEvent.contentIndex,
        partial,
        type: "toolcall_start",
      };

    case "toolcall_delta": {
      const content = partial.content[proxyEvent.contentIndex];
      if (content?.type === "toolCall") {
        (content as any).partialJson += proxyEvent.delta;
        content.arguments =
          parseStreamingJson((content as any).partialJson) || {};
        partial.content[proxyEvent.contentIndex] = { ...content }; // Trigger reactivity
        return {
          contentIndex: proxyEvent.contentIndex,
          delta: proxyEvent.delta,
          partial,
          type: "toolcall_delta",
        };
      }
      throw new Error("Received toolcall_delta for non-toolCall content");
    }

    case "toolcall_end": {
      const content = partial.content[proxyEvent.contentIndex];
      if (content?.type === "toolCall") {
        delete (content as any).partialJson;
        return {
          contentIndex: proxyEvent.contentIndex,
          partial,
          toolCall: content,
          type: "toolcall_end",
        };
      }
      return undefined;
    }

    default: {
      const _exhaustiveCheck: never = proxyEvent;
      console.warn(`Unhandled proxy event type: ${(proxyEvent as any).type}`);
      return undefined;
    }
  }
}
