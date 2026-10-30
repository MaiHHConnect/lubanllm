/** Claude Messages wire format over the local OpenAI-compatible inference path. */
import { parseJSON, stringifyJSON } from "./wire.ts";

const object = (value: any): value is Record<string, any> => value !== null && typeof value === "object" && !Array.isArray(value) && !(value instanceof Number);
export const claudePath = (path: string) => path === "/v1/messages" || path.startsWith("/v1/messages/");

export function claudeError(status: number, message: string) {
  const type = status === 401 ? "authentication_error" : status === 403 ? "permission_error"
    : status === 404 ? "not_found_error" : status === 413 ? "request_too_large"
    : status === 429 ? "rate_limit_error" : status >= 500 ? "api_error" : "invalid_request_error";
  return new Response(JSON.stringify({ type: "error", error: { type, message } }), {
    status, headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}

function cacheControl(value: any) {
  // These are cache scheduling hints. Accept them for SDK compatibility without
  // promising retention, a cache hit, or fabricated cache token accounting.
  if (value !== undefined && (!object(value) || value.type !== "ephemeral"
      || Object.keys(value).some(key => !["type", "ttl"].includes(key))
      || value.ttl !== undefined && !["5m", "1h"].includes(value.ttl))) {
    throw new TypeError("cache_control 仅支持 ephemeral 与 5m/1h ttl / Invalid cache_control");
  }
}
function textContent(value: any, allowEmpty = false): string {
  if (typeof value === "string") return value;
  if (!Array.isArray(value) || (!allowEmpty && !value.length) || value.length > 256) throw new TypeError("content 需要文字或文本部分 / Text content required");
  return value.map(part => {
    if (!object(part) || part.type !== "text" || typeof part.text !== "string"
        || Object.keys(part).some(key => !["type", "text", "cache_control"].includes(key))) {
      throw new TypeError("不支持媒体或非文本结果 / Media and non-text results are not supported");
    }
    cacheControl(part.cache_control);
    return part.text;
  }).join("");
}
const toolName = (value: any) => typeof value === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(value);
const callID = (value: any) => typeof value === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(value);

export function claudeRequest(body: any, version: string | null) {
  if (version !== "2023-06-01") throw new TypeError("anthropic-version 需要为 2023-06-01 / Unsupported or missing anthropic-version");
  if (!object(body)) throw new TypeError("请求需要 JSON 对象 / JSON object required");
  const allowed = ["model", "max_tokens", "messages", "system", "stream", "temperature", "metadata", "tools", "tool_choice", "thinking", "stop_sequences"];
  if (Object.keys(body).some(key => !allowed.includes(key))) throw new TypeError("请求包含尚不支持的参数 / Unsupported Messages parameter");
  if (body.model !== "luban0") throw new TypeError("模型名称应为 luban0 / Model must be luban0");
  if (!Number.isSafeInteger(body.max_tokens) || body.max_tokens < 1 || body.max_tokens > 32768) throw new TypeError("max_tokens 应为 1–32768 的整数 / Invalid max_tokens");
  if (body.stream !== undefined && typeof body.stream !== "boolean") throw new TypeError("stream 需要布尔值 / Invalid stream");
  if (body.temperature !== undefined && (!(typeof body.temperature === "number" || body.temperature instanceof Number)
      || !Number.isFinite(Number(body.temperature)) || Number(body.temperature) < 0 || Number(body.temperature) > 1)) throw new TypeError("temperature 应为 0–1 / Invalid temperature");
  if (body.tools !== undefined && (!Array.isArray(body.tools) || body.tools.length > 128)) throw new TypeError("tools 最多 128 项 / Invalid tools");
  const names = new Set<string>();
  const tools = (body.tools ?? []).map((tool: any) => {
    if (!object(tool) || Object.keys(tool).some(key => !["name", "description", "input_schema", "cache_control", "eager_input_streaming"].includes(key))
        || !toolName(tool.name) || names.has(tool.name) || !object(tool.input_schema) || tool.input_schema.type !== "object"
        || tool.description !== undefined && typeof tool.description !== "string"
        || tool.eager_input_streaming !== undefined && typeof tool.eager_input_streaming !== "boolean") throw new TypeError("工具定义无效 / Invalid tool definition");
    cacheControl(tool.cache_control); names.add(tool.name);
    return { type: "function", function: { name: tool.name, parameters: tool.input_schema,
      ...(tool.description === undefined ? {} : { description: tool.description }) } };
  });
  let choice: any, parallel: boolean | undefined;
  if (body.tool_choice !== undefined) {
    const selected = body.tool_choice;
    if (!object(selected) || !["auto", "any", "tool", "none"].includes(selected.type)
        || Object.keys(selected).some(key => !["type", "name", "disable_parallel_tool_use"].includes(key))
        || selected.type !== "tool" && selected.name !== undefined
        || selected.disable_parallel_tool_use !== undefined && typeof selected.disable_parallel_tool_use !== "boolean"
        || selected.type !== "none" && !tools.length
        || selected.type === "tool" && !names.has(selected.name)) throw new TypeError("tool_choice 无效 / Invalid tool_choice");
    choice = selected.type === "tool" ? { type: "function", function: { name: selected.name } }
      : selected.type === "any" ? "required" : selected.type;
    if (selected.disable_parallel_tool_use !== undefined) parallel = !selected.disable_parallel_tool_use;
  }
  if (body.stop_sequences !== undefined && (!Array.isArray(body.stop_sequences) || body.stop_sequences.length)) throw new TypeError("暂不支持 stop_sequences / Custom stop sequences are not supported");
  if (body.thinking !== undefined && (!object(body.thinking) || body.thinking.type !== "disabled"
      || Object.keys(body.thinking).length !== 1)) throw new TypeError("暂不支持 Claude 扩展思考 / Extended thinking is not supported");
  if (body.metadata !== undefined && (!object(body.metadata) || Object.keys(body.metadata).some(key => key !== "user_id")
      || body.metadata.user_id !== undefined && body.metadata.user_id !== null && typeof body.metadata.user_id !== "string")) throw new TypeError("metadata 无效 / Invalid metadata");
  if (!Array.isArray(body.messages) || !body.messages.length) throw new TypeError("messages 需要非空对话消息数组 / Invalid messages");
  const messages: any[] = [], pending = new Set<string>(), used = new Set<string>();
  if (body.system !== undefined) {
    const system = textContent(body.system);
    if (system.trim()) messages.push({ role: "system", content: system });
  }
  for (const message of body.messages) {
    if (!object(message) || !["user", "assistant"].includes(message.role)
        || Object.keys(message).some(key => !["role", "content"].includes(key))) throw new TypeError("消息只支持 user/assistant / Only user and assistant messages are supported");
    const parts = typeof message.content === "string" ? [{ type: "text", text: message.content }] : message.content;
    if (!Array.isArray(parts) || !parts.length || parts.length > 256) throw new TypeError("消息内容无效 / Invalid message content");
    if (message.role === "assistant" && pending.size) throw new TypeError("缺少工具结果 / Missing tool results");
    let content = "", sawText = false;
    const calls: any[] = [];
    for (const part of parts) {
      if (!object(part)) throw new TypeError("消息内容无效 / Invalid message content");
      if (part.type === "text") {
        // Chat Completions places assistant text before tool calls and tool results before user text.
        if (calls.length || pending.size) throw new TypeError("文本须在工具调用前，工具结果须在用户文本前 / Invalid tool/text ordering");
        content += textContent([part]); sawText = true;
      } else if (part.type === "tool_use" && message.role === "assistant") {
        if (Object.keys(part).some(key => !["type", "id", "name", "input", "cache_control"].includes(key))
            || !callID(part.id) || used.has(part.id) || !toolName(part.name) || !object(part.input)) throw new TypeError("tool_use 无效 / Invalid tool_use");
        cacheControl(part.cache_control); pending.add(part.id); used.add(part.id);
        calls.push({ id: part.id, type: "function", function: { name: part.name, arguments: stringifyJSON(part.input) } });
      } else if (part.type === "tool_result" && message.role === "user") {
        if (sawText || Object.keys(part).some(key => !["type", "tool_use_id", "content", "is_error", "cache_control"].includes(key))
            || !pending.has(part.tool_use_id) || part.is_error !== undefined && typeof part.is_error !== "boolean") throw new TypeError("tool_result 无效或没有对应调用 / Invalid or unmatched tool_result");
        cacheControl(part.cache_control); pending.delete(part.tool_use_id);
        messages.push({ role: "tool", tool_call_id: part.tool_use_id,
          content: (part.is_error ? "[Tool error]\n" : "") + (part.content === undefined ? "" : textContent(part.content, true)) });
      } else throw new TypeError("不支持该内容类型 / Unsupported content block");
    }
    if (content.trim() || calls.length) {
      const previous = messages.at(-1);
      if (!calls.length && previous?.role === message.role && !previous.tool_calls) previous.content += "\n" + content;
      else messages.push({ role: message.role, content: content || null, ...(calls.length ? { tool_calls: calls } : {}) });
    } else if (!parts.some((part: any) => part.type === "tool_result")) throw new TypeError("消息 content 需要非空文本或工具调用 / Non-empty text or tool call required");
  }
  if (body.messages.at(-1)?.role === "assistant") throw new TypeError("暂不支持助手预填，请以 user 消息结束 / Assistant prefill is not supported; end with a user message");
  if (pending.size) throw new TypeError("缺少工具结果 / Missing tool results");
  return { model: "luban0", messages, max_tokens: body.max_tokens, stream: body.stream ?? false,
    thinking: "off", ...(body.temperature === undefined ? {} : { temperature: body.temperature }),
    ...(tools.length ? { tools } : {}), ...(choice === undefined ? {} : { tool_choice: choice }),
    ...(parallel === undefined ? {} : { parallel_tool_calls: parallel }) };
}

function usage(value: any) {
  if (!object(value) || !Number.isSafeInteger(value.prompt_tokens) || value.prompt_tokens < 0
      || !Number.isSafeInteger(value.completion_tokens) || value.completion_tokens < 0) throw new Error("模型未返回有效用量 / Missing valid model usage");
  const nested = value.prompt_tokens_details?.cached_tokens;
  const cached = nested === undefined ? value.cached_prompt_tokens : nested;
  if (cached !== undefined && (!Number.isSafeInteger(cached) || cached < 0 || cached > value.prompt_tokens)) {
    throw new Error("模型返回的缓存用量无效 / Invalid model cache usage");
  }
  return { input_tokens: value.prompt_tokens - (cached ?? 0), output_tokens: value.completion_tokens,
    ...(cached === undefined ? {} : { cache_read_input_tokens: cached, cache_creation_input_tokens: 0 }) };
}
function stopReason(value: any) {
  if (!["stop", "length", "tool_calls"].includes(value)) throw new Error("模型未完整结束 / Incomplete model response");
  return value === "tool_calls" ? "tool_use" : value === "length" ? "max_tokens" : "end_turn";
}
function message(id: string, content: any[], stop_reason: string | null, counts: any) {
  return { id, type: "message", role: "assistant", model: "luban0", content, stop_reason, stop_sequence: null, usage: counts };
}

function responseTools(value: any, finish: string): any[] {
  if (value === undefined) value = [];
  if (!Array.isArray(value) || value.length > 128 || (value.length > 0) !== (finish === "tool_use")) throw new Error("模型工具结束状态无效 / Invalid tool finish state");
  const seen = new Set<string>();
  return value.map(call => {
    if (!object(call) || call.type !== "function" || !callID(call.id) || seen.has(call.id)
        || !object(call.function) || !toolName(call.function.name) || typeof call.function.arguments !== "string") throw new Error("模型工具调用无效 / Invalid model tool call");
    const input = parseJSON(call.function.arguments);
    if (!object(input)) throw new Error("工具参数须为 JSON 对象 / Tool arguments must be a JSON object");
    seen.add(call.id);
    return { type: "tool_use", id: call.id, name: call.function.name, input };
  });
}

export async function claudeResponse(response: Response, stream: boolean, mask: (error: unknown) => string): Promise<Response> {
  if (!response.ok) {
    let value: any;
    try { value = await response.json(); } catch { /* Preserve status without leaking an upstream HTML response. */ }
    return claudeError(response.status, mask(new Error(value?.error?.message ?? value?.detail ?? "模型请求失败 / Model request failed")));
  }
  const id = "msg_" + crypto.randomUUID().replaceAll("-", "");
  if (!stream) {
    try {
      const value = await response.json() as any, choice = value?.choices?.[0];
      const finish = stopReason(choice?.finish_reason), content: any[] = [];
      if (choice.message?.content !== null && typeof choice.message?.content !== "string") throw new Error("模型回复格式无效 / Invalid model response");
      if (choice.message.content) content.push({ type: "text", text: choice.message.content });
      content.push(...responseTools(choice.message.tool_calls, finish));
      return new Response(stringifyJSON(message(id, content, finish, usage(value.usage))), { headers: { "Content-Type": "application/json", "Cache-Control": "no-store" } });
    } catch (error) { return claudeError(502, mask(error)); }
  }
  if (!response.body) return claudeError(502, "模型未返回流 / Missing model stream");
  const reader = response.body.getReader(), encoder = new TextEncoder();
  const frame = (type: string, value: any) => encoder.encode(`event: ${type}\ndata: ${stringifyJSON({ type, ...value })}\n\n`);
  async function* events() {
    let complete = false;
    try {
      // Counts become available on the existing compute path's final event.
      // message_delta supplies cumulative counts, as supported by the Claude SDK.
      yield frame("message_start", { message: message(id, [], null, { input_tokens: 0, output_tokens: 0 }) });
      const decoder = new TextDecoder("utf-8", { fatal: true });
      let pending = "", data: string[] = [], finish: string | null = null, counts: ReturnType<typeof usage> | undefined;
      let textStarted = false, toolBytes = 0;
      const tools = new Map<number, { id: string; type: string; function: { name: string; arguments: string } }>();
      while (!complete) {
        const chunk = await reader.read();
        pending += chunk.done ? decoder.decode() : decoder.decode(chunk.value, { stream: true });
        if (pending.length > 8 * 1024 * 1024) throw new Error("模型流事件过大 / Model stream event too large");
        let newline: number;
        while ((newline = pending.indexOf("\n")) >= 0) {
          const line = pending.slice(0, newline).replace(/\r$/, ""); pending = pending.slice(newline + 1);
          if (line.startsWith(":")) { yield frame("ping", {}); continue; }
          if (line.startsWith("data:")) { data.push(line.slice(5).trimStart()); continue; }
          if (line || !data.length) continue;
          const payload = data.join("\n"); data = [];
          if (payload === "[DONE]") {
            if (!finish || !counts) throw new Error("模型流缺少结束状态或用量 / Incomplete model stream");
            const indexes = [...tools.keys()].sort((a, b) => a - b);
            if (indexes.some((index, offset) => index !== offset)) throw new Error("模型工具索引无效 / Invalid tool indexes");
            const blocks = responseTools(indexes.map(index => tools.get(index)), finish);
            if (textStarted) yield frame("content_block_stop", { index: 0 });
            // Validate the completed JSON before exposing a tool request to a client.
            for (let offset = 0; offset < blocks.length; offset++) {
              const block = blocks[offset], index = offset + Number(textStarted);
              yield frame("content_block_start", { index, content_block: { type: "tool_use", id: block.id, name: block.name, input: {} } });
              yield frame("content_block_delta", { index, delta: { type: "input_json_delta", partial_json: tools.get(offset)!.function.arguments } });
              yield frame("content_block_stop", { index });
            }
            yield frame("message_delta", { delta: { stop_reason: finish, stop_sequence: null }, usage: counts });
            yield frame("message_stop", {}); complete = true; break;
          }
          const value = parseJSON(payload), choice = value?.choices?.[0];
          if (value?.error) throw new Error(value.error.message ?? "模型生成失败 / Model generation failed");
          if (finish && (choice?.delta?.content || choice?.delta?.tool_calls?.length || choice?.finish_reason)) throw new Error("模型结束后仍返回内容 / Invalid model stream order");
          if (choice?.delta?.content !== undefined) {
            if (choice.delta.content !== null && typeof choice.delta.content !== "string") throw new Error("模型流文字无效 / Invalid stream text");
            if (choice.delta.content) {
              if (tools.size) throw new Error("模型工具调用后仍返回文字 / Invalid tool/text stream order");
              if (!textStarted) { textStarted = true; yield frame("content_block_start", { index: 0, content_block: { type: "text", text: "" } }); }
              yield frame("content_block_delta", { index: 0, delta: { type: "text_delta", text: choice.delta.content } });
            }
          }
          if (choice?.delta?.tool_calls !== undefined) {
            if (!Array.isArray(choice.delta.tool_calls)) throw new Error("模型工具流无效 / Invalid tool stream");
            for (const part of choice.delta.tool_calls) {
              if (!object(part) || !Number.isSafeInteger(part.index) || part.index < 0 || part.index >= 128
                  || part.type !== undefined && part.type !== "function"
                  || part.id !== undefined && typeof part.id !== "string"
                  || part.function !== undefined && (!object(part.function)
                    || part.function.name !== undefined && typeof part.function.name !== "string"
                    || part.function.arguments !== undefined && typeof part.function.arguments !== "string")) throw new Error("模型工具流无效 / Invalid tool stream");
              const call = tools.get(part.index) ?? { id: "", type: "function", function: { name: "", arguments: "" } };
              call.id += part.id ?? ""; call.function.name += part.function?.name ?? "";
              call.function.arguments += part.function?.arguments ?? "";
              toolBytes += (part.id?.length ?? 0) + (part.function?.name?.length ?? 0) + (part.function?.arguments?.length ?? 0);
              if (toolBytes > 8 * 1024 * 1024) throw new Error("模型工具流过大 / Model tool stream too large");
              tools.set(part.index, call);
            }
          }
          if (choice?.finish_reason !== undefined && choice.finish_reason !== null) finish = stopReason(choice.finish_reason);
          if (value?.usage) counts = usage(value.usage);
        }
        if (chunk.done && !complete) throw new Error("模型流意外中断 / Model stream ended unexpectedly");
      }
    } catch (error) {
      yield frame("error", { error: { type: "api_error", message: mask(error) } });
    } finally {
      try { await reader.cancel(); } catch { /* Disconnects already cancel upstream. */ }
      reader.releaseLock();
    }
  }
  const iterator = events();
  return new Response(new ReadableStream<Uint8Array>({
    async pull(controller) {
      const next = await iterator.next();
      if (next.done) controller.close(); else controller.enqueue(next.value);
    },
    async cancel() {
      // Cancel the pending read first, so return() cannot wait on a stalled model.
      try { await reader.cancel(); } catch {}
      await iterator.return(undefined);
    },
  }), { headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-store", "X-Accel-Buffering": "no" } });
}
