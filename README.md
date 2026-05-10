# @shiit/agent-core

---

## WARNING!!! THIS IS A HIGH EFFORT SLOP FORK, USE AT YOUR OWN RISK

Fork of `@mariozechner/pi-agent-core` with streaming message ID support.

## Why This Fork

The upstream emits `message_start`, `message_update`, and `message_end` events without IDs, making it impossible for clients to match streaming updates to specific messages.

This fork adds:

- `id: string` field to `message_start`, `message_update`, and `message_end` events
- `delta: string` and `thinkingDelta: string` fields to `message_update` for efficient streaming

## Installation

```bash
npm install @shiit/agent-core
```

## Usage

```typescript
import { agentLoop } from "@shiit/agent-core";

// Events now include message IDs
for await (const event of agentLoop(prompts, context, config)) {
  switch (event.type) {
    case "message_start":
      console.log(event.id); // Unique message ID
      break;
    case "message_update":
      console.log(event.id, event.delta, event.thinkingDelta);
      break;
    case "message_end":
      console.log(event.id);
      break;
  }
}
```

## Changes from Upstream

### Added Fields

All `AgentEvent` types that were missing `id` now include it:

| Event            | Added Fields                                           |
| ---------------- | ------------------------------------------------------ |
| `message_start`  | `id: string`                                           |
| `message_update` | `id: string`, `delta: string`, `thinkingDelta: string` |
| `message_end`    | `id: string`                                           |

### Dependencies

- Uses `@shiit/id` for ID generation (Crockford base32 nanoid)

## Version

Version `0.66.1` matches upstream `@mariozechner/pi-agent-core@0.66.1`.

## Upstream

- Repository: https://github.com/badlogic/pi-mono
- Version: v0.66.1

## License

MIT
