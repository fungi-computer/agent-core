import {
  type ImageContent,
  type Message,
  type Model,
  type SimpleStreamOptions,
  streamSimple,
  type TextContent,
  type ThinkingBudgets,
  type Transport,
} from "@earendil-works/pi-ai/compat";

import type {
  AfterToolCallContext,
  AfterToolCallResult,
  AgentContext,
  AgentEvent,
  AgentLoopConfig,
  AgentMessage,
  AgentState,
  AgentTool,
  BeforeToolCallContext,
  BeforeToolCallResult,
  StreamFn,
  ToolExecutionMode,
} from "./types.js";

import { runAgentLoop, runAgentLoopContinue } from "./agent-loop.js";

function defaultConvertToLlm(messages: AgentMessage[]): Message[] {
  return messages.filter(
    (message) =>
      message.role === "user" ||
      message.role === "assistant" ||
      message.role === "toolResult",
  );
}

const EMPTY_USAGE = {
  cacheRead: 0,
  cacheWrite: 0,
  cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0, total: 0 },
  input: 0,
  output: 0,
  totalTokens: 0,
};

const DEFAULT_MODEL = {
  api: "unknown",
  baseUrl: "",
  contextWindow: 0,
  cost: { cacheRead: 0, cacheWrite: 0, input: 0, output: 0 },
  id: "unknown",
  input: [],
  maxTokens: 0,
  name: "unknown",
  provider: "unknown",
  reasoning: false,
} satisfies Model<any>;

/** Options for constructing an {@link Agent}. */
export interface AgentOptions {
  afterToolCall?: (
    context: AfterToolCallContext,
    signal?: AbortSignal,
  ) => Promise<AfterToolCallResult | undefined>;
  beforeToolCall?: (
    context: BeforeToolCallContext,
    signal?: AbortSignal,
  ) => Promise<BeforeToolCallResult | undefined>;
  convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
  followUpMode?: QueueMode;
  getApiKey?: (
    provider: string,
  ) => Promise<string | undefined> | string | undefined;
  initialState?: Partial<
    Omit<
      AgentState,
      "errorMessage" | "isStreaming" | "pendingToolCalls" | "streamingMessage"
    >
  >;
  maxRetryDelayMs?: number;
  onPayload?: SimpleStreamOptions["onPayload"];
  sessionId?: string;
  steeringMode?: QueueMode;
  streamFn?: StreamFn;
  thinkingBudgets?: ThinkingBudgets;
  toolExecution?: ToolExecutionMode;
  transformContext?: (
    messages: AgentMessage[],
    signal?: AbortSignal,
  ) => Promise<AgentMessage[]>;
  transport?: Transport;
}

interface ActiveRun {
  abortController: AbortController;
  promise: Promise<void>;
  resolve: () => void;
}

type MutableAgentState = {
  errorMessage?: string;
  isStreaming: boolean;
  pendingToolCalls: Set<string>;
  streamingMessage?: AgentMessage;
} & Omit<
  AgentState,
  "errorMessage" | "isStreaming" | "pendingToolCalls" | "streamingMessage"
>;

type QueueMode = "all" | "one-at-a-time";

class PendingMessageQueue {
  private messages: AgentMessage[] = [];

  constructor(public mode: QueueMode) {}

  clear(): void {
    this.messages = [];
  }

  drain(): AgentMessage[] {
    if (this.mode === "all") {
      const drained = this.messages.slice();
      this.messages = [];
      return drained;
    }

    const first = this.messages[0];
    if (!first) {
      return [];
    }
    this.messages = this.messages.slice(1);
    return [first];
  }

  enqueue(message: AgentMessage): void {
    this.messages.push(message);
  }

  hasItems(): boolean {
    return this.messages.length > 0;
  }
}

/**
 * Stateful wrapper around the low-level agent loop.
 *
 * `Agent` owns the current transcript, emits lifecycle events, executes tools,
 * and exposes queueing APIs for steering and follow-up messages.
 */
export class Agent {
  public afterToolCall?: (
    context: AfterToolCallContext,
    signal?: AbortSignal,
  ) => Promise<AfterToolCallResult | undefined>;
  public beforeToolCall?: (
    context: BeforeToolCallContext,
    signal?: AbortSignal,
  ) => Promise<BeforeToolCallResult | undefined>;
  public convertToLlm: (
    messages: AgentMessage[],
  ) => Message[] | Promise<Message[]>;
  public getApiKey?: (
    provider: string,
  ) => Promise<string | undefined> | string | undefined;

  /** Optional cap for provider-requested retry delays. */
  public maxRetryDelayMs?: number;
  public onPayload?: SimpleStreamOptions["onPayload"];
  /** Session identifier forwarded to providers for cache-aware backends. */
  public sessionId?: string;
  public streamFn: StreamFn;
  /** Optional per-level thinking token budgets forwarded to the stream function. */
  public thinkingBudgets?: ThinkingBudgets;
  /** Tool execution strategy for assistant messages that contain multiple tool calls. */
  public toolExecution: ToolExecutionMode;
  public transformContext?: (
    messages: AgentMessage[],
    signal?: AbortSignal,
  ) => Promise<AgentMessage[]>;
  /** Preferred transport forwarded to the stream function. */
  public transport: Transport;
  /** Controls how queued follow-up messages are drained. */
  set followUpMode(mode: QueueMode) {
    this.followUpQueue.mode = mode;
  }
  get followUpMode(): QueueMode {
    return this.followUpQueue.mode;
  }
  /** Active abort signal for the current run, if any. */
  get signal(): AbortSignal | undefined {
    return this.activeRun?.abortController.signal;
  }
  /**
   * Current agent state.
   *
   * Assigning `state.tools` or `state.messages` copies the provided top-level array.
   */
  get state(): AgentState {
    return this._state;
  }
  /** Controls how queued steering messages are drained. */
  set steeringMode(mode: QueueMode) {
    this.steeringQueue.mode = mode;
  }

  get steeringMode(): QueueMode {
    return this.steeringQueue.mode;
  }

  private _state: MutableAgentState;

  private activeRun?: ActiveRun;
  private runStartTime?: number;

  private readonly followUpQueue: PendingMessageQueue;

  private readonly listeners = new Set<
    (event: AgentEvent, signal: AbortSignal) => Promise<void> | void
  >();

  private readonly steeringQueue: PendingMessageQueue;

  constructor(options: AgentOptions = {}) {
    this._state = createMutableAgentState(options.initialState);
    this.convertToLlm = options.convertToLlm ?? defaultConvertToLlm;
    this.transformContext = options.transformContext;
    this.streamFn = options.streamFn ?? streamSimple;
    this.getApiKey = options.getApiKey;
    this.onPayload = options.onPayload;
    this.beforeToolCall = options.beforeToolCall;
    this.afterToolCall = options.afterToolCall;
    this.steeringQueue = new PendingMessageQueue(
      options.steeringMode ?? "one-at-a-time",
    );
    this.followUpQueue = new PendingMessageQueue(
      options.followUpMode ?? "one-at-a-time",
    );
    this.sessionId = options.sessionId;
    this.thinkingBudgets = options.thinkingBudgets;
    this.transport = options.transport ?? "sse";
    this.maxRetryDelayMs = options.maxRetryDelayMs;
    this.toolExecution = options.toolExecution ?? "parallel";
  }

  /** Abort the current run, if one is active. */
  abort(): void {
    this.activeRun?.abortController.abort();
  }

  /** Remove all queued steering and follow-up messages. */
  clearAllQueues(): void {
    this.clearSteeringQueue();
    this.clearFollowUpQueue();
  }

  /** Remove all queued follow-up messages. */
  clearFollowUpQueue(): void {
    this.followUpQueue.clear();
  }

  /** Remove all queued steering messages. */
  clearSteeringQueue(): void {
    this.steeringQueue.clear();
  }

  /** Continue from the current transcript. The last message must be a user or tool-result message. */
  async continue(): Promise<void> {
    if (this.activeRun) {
      throw new Error(
        "Agent is already processing. Wait for completion before continuing.",
      );
    }

    const lastMessage = this._state.messages[this._state.messages.length - 1];
    if (!lastMessage) {
      throw new Error("No messages to continue from");
    }

    if (lastMessage.role === "assistant") {
      const queuedSteering = this.steeringQueue.drain();
      if (queuedSteering.length > 0) {
        await this.runPromptMessages(queuedSteering, {
          skipInitialSteeringPoll: true,
        });
        return;
      }

      const queuedFollowUps = this.followUpQueue.drain();
      if (queuedFollowUps.length > 0) {
        await this.runPromptMessages(queuedFollowUps);
        return;
      }

      throw new Error("Cannot continue from message role: assistant");
    }

    await this.runContinuation();
  }

  /** Queue a message to run only after the agent would otherwise stop. */
  followUp(message: AgentMessage): void {
    this.followUpQueue.enqueue(message);
  }

  /** Returns true when either queue still contains pending messages. */
  hasQueuedMessages(): boolean {
    return this.steeringQueue.hasItems() || this.followUpQueue.hasItems();
  }

  /** Start a new prompt from text, a single message, or a batch of messages. */
  async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
  async prompt(input: string, images?: ImageContent[]): Promise<void>;
  async prompt(
    input: AgentMessage | AgentMessage[] | string,
    images?: ImageContent[],
  ): Promise<void> {
    if (this.activeRun) {
      throw new Error(
        "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
      );
    }
    const messages = this.normalizePromptInput(input, images);
    await this.runPromptMessages(messages);
  }

  /** Clear transcript state, runtime state, and queued messages. */
  reset(): void {
    this._state.messages = [];
    this._state.isStreaming = false;
    this._state.streamingMessage = undefined;
    this._state.pendingToolCalls = new Set<string>();
    this._state.errorMessage = undefined;
    this.clearFollowUpQueue();
    this.clearSteeringQueue();
  }
  /** Queue a message to be injected after the current assistant turn finishes. */
  steer(message: AgentMessage): void {
    this.steeringQueue.enqueue(message);
  }
  /**
   * Subscribe to agent lifecycle events.
   *
   * Listener promises are awaited in subscription order and are included in
   * the current run's settlement. Listeners also receive the active abort
   * signal for the current run.
   *
   * `agent_end` is the final emitted event for a run, but the agent does not
   * become idle until all awaited listeners for that event have settled.
   */
  subscribe(
    listener: (event: AgentEvent, signal: AbortSignal) => Promise<void> | void,
  ): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Resolve when the current run and all awaited event listeners have finished.
   *
   * This resolves after `agent_end` listeners settle.
   */
  waitForIdle(): Promise<void> {
    return this.activeRun?.promise ?? Promise.resolve();
  }

  private createContextSnapshot(): AgentContext {
    return {
      messages: this._state.messages.slice(),
      systemPrompt: this._state.systemPrompt,
      tools: this._state.tools.slice(),
    };
  }

  private createLoopConfig(
    options: { skipInitialSteeringPoll?: boolean } = {},
  ): AgentLoopConfig {
    let skipInitialSteeringPoll = options.skipInitialSteeringPoll === true;
    return {
      afterToolCall: this.afterToolCall,
      beforeToolCall: this.beforeToolCall,
      convertToLlm: this.convertToLlm,
      getApiKey: this.getApiKey,
      getFollowUpMessages: async () => this.followUpQueue.drain(),
      getSteeringMessages: async () => {
        if (skipInitialSteeringPoll) {
          skipInitialSteeringPoll = false;
          return [];
        }
        return this.steeringQueue.drain();
      },
      maxRetryDelayMs: this.maxRetryDelayMs,
      model: this._state.model,
      onPayload: this.onPayload,
      reasoning:
        this._state.thinkingLevel === "off"
          ? undefined
          : this._state.thinkingLevel,
      sessionId: this.sessionId,
      thinkingBudgets: this.thinkingBudgets,
      toolExecution: this.toolExecution,
      transformContext: this.transformContext,
      transport: this.transport,
    };
  }

  private finishRun(): void {
    this._state.isStreaming = false;
    this._state.streamingMessage = undefined;
    this._state.pendingToolCalls = new Set<string>();
    this.activeRun?.resolve();
    this.activeRun = undefined;
  }

  private async handleRunFailure(
    error: unknown,
    aborted: boolean,
  ): Promise<void> {
    console.error("[handleRunFailure] Error:", error);
    const failureMessage = {
      api: this._state.model.api,
      content: [{ text: "", type: "text" }],
      errorMessage: error instanceof Error ? error.message : String(error),
      model: this._state.model.id,
      provider: this._state.model.provider,
      role: "assistant",
      stopReason: aborted ? "aborted" : "error",
      timestamp: Date.now(),
      usage: EMPTY_USAGE,
    } satisfies AgentMessage;
    this._state.messages.push(failureMessage);
    this._state.errorMessage = failureMessage.errorMessage;
    await this.processEvents({ messages: [failureMessage], type: "agent_end" });
  }

  private normalizePromptInput(
    input: AgentMessage | AgentMessage[] | string,
    images?: ImageContent[],
  ): AgentMessage[] {
    if (Array.isArray(input)) {
      return input;
    }

    if (typeof input !== "string") {
      return [input];
    }

    const content: (ImageContent | TextContent)[] = [
      { text: input, type: "text" },
    ];
    if (images && images.length > 0) {
      content.push(...images);
    }
    return [{ content, role: "user", timestamp: Date.now() }];
  }

  /**
   * Reduce internal state for a loop event, then await listeners.
   *
   * `agent_end` only means no further loop events will be emitted. The run is
   * considered idle later, after all awaited listeners for `agent_end` finish
   * and `finishRun()` clears runtime-owned state.
   */
  private async processEvents(event: AgentEvent): Promise<void> {
    if (this.runStartTime && event.type !== "agent_end") {
      const elapsed = Date.now() - this.runStartTime;
      console.log(`[Agent] first event: ${event.type} after ${elapsed}ms`);
      this.runStartTime = undefined;
    }
    switch (event.type) {
      case "agent_end":
        this._state.streamingMessage = undefined;
        break;

      case "message_end":
        this._state.streamingMessage = undefined;
        this._state.messages.push(event.message);
        break;

      case "message_start":
        this._state.streamingMessage = event.message;
        break;

      case "message_update":
        this._state.streamingMessage = event.message;
        break;

      case "tool_execution_end": {
        const pendingToolCalls = new Set(this._state.pendingToolCalls);
        pendingToolCalls.delete(event.toolCallId);
        this._state.pendingToolCalls = pendingToolCalls;
        break;
      }

      case "tool_execution_start": {
        const pendingToolCalls = new Set(this._state.pendingToolCalls);
        pendingToolCalls.add(event.toolCallId);
        this._state.pendingToolCalls = pendingToolCalls;
        break;
      }

      case "turn_end":
        if (event.message.role === "assistant" && event.message.errorMessage) {
          this._state.errorMessage = event.message.errorMessage;
        }
        break;
    }

    const signal = this.activeRun?.abortController.signal;
    if (!signal) {
      throw new Error("Agent listener invoked outside active run");
    }
    for (const listener of this.listeners) {
      await listener(event, signal);
    }
  }

  private async runContinuation(): Promise<void> {
    await this.runWithLifecycle(async (signal) => {
      await runAgentLoopContinue(
        this.createContextSnapshot(),
        this.createLoopConfig(),
        (event) => this.processEvents(event),
        signal,
        this.streamFn,
      );
    });
  }

  private async runPromptMessages(
    messages: AgentMessage[],
    options: { skipInitialSteeringPoll?: boolean } = {},
  ): Promise<void> {
    await this.runWithLifecycle(async (signal) => {
      await runAgentLoop(
        messages,
        this.createContextSnapshot(),
        this.createLoopConfig(options),
        (event) => this.processEvents(event),
        signal,
        this.streamFn,
      );
    });
  }

  private async runWithLifecycle(
    executor: (signal: AbortSignal) => Promise<void>,
  ): Promise<void> {
    if (this.activeRun) {
      throw new Error("Agent is already processing.");
    }

    const abortController = new AbortController();
    let resolvePromise = () => {};
    const promise = new Promise<void>((resolve) => {
      resolvePromise = resolve;
    });
    this.activeRun = { abortController, promise, resolve: resolvePromise };
    this.runStartTime = Date.now();

    this._state.isStreaming = true;
    this._state.streamingMessage = undefined;
    this._state.errorMessage = undefined;

    try {
      await executor(abortController.signal);
    } catch (error) {
      await this.handleRunFailure(error, abortController.signal.aborted);
    } finally {
      this.finishRun();
    }
  }
}

function createMutableAgentState(
  initialState?: Partial<
    Omit<
      AgentState,
      "errorMessage" | "isStreaming" | "pendingToolCalls" | "streamingMessage"
    >
  >,
): MutableAgentState {
  let tools = initialState?.tools?.slice() ?? [];
  let messages = initialState?.messages?.slice() ?? [];

  return {
    errorMessage: undefined,
    isStreaming: false,
    get messages() {
      return messages;
    },
    set messages(nextMessages: AgentMessage[]) {
      messages = nextMessages.slice();
    },
    model: initialState?.model ?? DEFAULT_MODEL,
    pendingToolCalls: new Set<string>(),
    streamingMessage: undefined,
    systemPrompt: initialState?.systemPrompt ?? "",
    thinkingLevel: initialState?.thinkingLevel ?? "off",
    get tools() {
      return tools;
    },
    set tools(nextTools: AgentTool<any>[]) {
      tools = nextTools.slice();
    },
  };
}
