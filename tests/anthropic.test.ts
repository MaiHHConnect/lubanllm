import { describe, expect, test } from "bun:test";
import { claudeRequest, claudeResponse } from "../backend/anthropic.ts";
import { parseJSON } from "../backend/wire.ts";

const payload = { model: "luban0", max_tokens: 32, messages: [{ role: "user", content: "你好" }] };
const mask = (error: any) => String(error?.message ?? error).replaceAll("private-key", "[redacted]");
const encode = (value: any) => "data: " + JSON.stringify(value) + "\n\n";
const delta = (text: string) => encode({ choices: [{ delta: { content: text }, finish_reason: null }] });
const final = (reason = "stop") => encode({ choices: [{ delta: {}, finish_reason: reason }], usage: {
  prompt_tokens: 7, completion_tokens: 3,
} });
const frames = (text: string) => text.split("\n\n").filter(Boolean).map(frame => JSON.parse(frame.split("\n")[1]!.slice(6)));

describe("Claude text protocol", () => {
  test("forwards upstream heartbeat comments as ping without model content or usage", async () => {
    const response = await claudeResponse(new Response(": keepalive\n\n" + delta("答案") + ": keepalive\n\n" + final() + "data: [DONE]\n\n"), true, mask);
    const values = frames(await response.text());
    expect(values.filter(v => v.type === "ping")).toEqual([{ type: "ping" }, { type: "ping" }]);
    expect(values.filter(v => v.type === "content_block_delta").map(v => v.delta.text).join("")).toBe("答案");
    expect(values.find(v => v.type === "message_delta").usage).toEqual({ input_tokens: 7, output_tokens: 3 });
    expect(values.at(-1).type).toBe("message_stop");
  });
  test("preserves system and multipart text and maps only supported generation parameters", () => {
    const input = parseJSON(JSON.stringify({ ...payload, temperature: .6, stream: true, system: [
      { type: "text", text: "规则甲" }, { type: "text", text: "规则乙" },
    ], thinking: { type: "disabled" }, tools: [], metadata: { user_id: "opaque" } }));
    const converted = claudeRequest(input, "2023-06-01");
    expect(Number(converted.temperature)).toBe(.6);
    expect(converted).toMatchObject({ model: "luban0", max_tokens: 32, stream: true, thinking: "off",
      messages: [{ role: "system", content: "规则甲规则乙" }, ...payload.messages] });
    expect(claudeRequest({ ...payload, messages: [{ role: "user", content: "甲" }, { role: "user", content: "乙" }] }, "2023-06-01").messages)
      .toEqual([{ role: "user", content: "甲\n乙" }]);
  });
  test("preserves more than 100 alternating messages and the system message", () => {
    const messages = Array.from({ length: 101 }, (_, index) => ({
      role: index % 2 ? "assistant" : "user", content: `消息${index}`,
    }));
    expect(claudeRequest({ ...payload, system: "规则", messages }, "2023-06-01").messages)
      .toEqual([{ role: "system", content: "规则" }, ...messages]);
  });
  test("still rejects missing, non-array and empty messages", () => {
    for (const messages of [undefined, null, {}, "history", []]) {
      expect(() => claudeRequest({ ...payload, messages }, "2023-06-01")).toThrow("Invalid messages");
    }
  });
  test("rejects unsupported semantics rather than discarding tool, media or sampling inputs", () => {
    for (const body of [null, [], { ...payload, tools: [{}] }, { ...payload, tool_choice: { type: "auto" } },
      { ...payload, messages: [{ role: "assistant", content: [{ type: "tool_use", id: "a", name: "x", input: {} }] }] },
      { ...payload, messages: [{ role: "user", content: [{ type: "image", source: {} }] }] },
      { ...payload, messages: [{ role: "assistant", content: "ok", tool_calls: [{}] }] },
      { ...payload, messages: [{ role: "user", content: " " }] }, { ...payload, max_tokens: true },
      { ...payload, messages: [{ role: "assistant", content: "prefill" }] },
      { ...payload, max_tokens: 0 }, { ...payload, max_tokens: 32769 }, { ...payload, model: "claude" },
      { ...payload, stream: 1 }, { ...payload, temperature: 1.1 }, { ...payload, temperature: null },
      { ...payload, thinking: { type: "enabled", budget_tokens: 10 } }, { ...payload, top_p: .9 },
      { ...payload, stop_sequences: ["STOP"] },
      { ...payload, system: [{ type: "text", text: "x", cache_control: { type: "persistent" } }] },
    ]) expect(() => claudeRequest(body, "2023-06-01")).toThrow();
    expect(() => claudeRequest(payload, null)).toThrow();
  });
  test("maps length stop and upstream errors without exposing credentials", async () => {
    const response = await claudeResponse(Response.json({ choices: [{ message: { content: "答案" }, finish_reason: "length" }],
      usage: { prompt_tokens: 7, completion_tokens: 3 } }), false, mask);
    expect(await response.json()).toMatchObject({ stop_reason: "max_tokens", usage: { input_tokens: 7, output_tokens: 3 } });
    const error = await claudeResponse(Response.json({ error: { message: "private-key failed" } }, { status: 409 }), false, mask);
    expect(error.status).toBe(409); expect(await error.text()).toContain("[redacted]");
    const missing = await claudeResponse(Response.json({ choices: [{ message: { content: "x" }, finish_reason: "stop" }] }), false, mask);
    expect(missing.status).toBe(502);
  });
  test("reports actual cached input once with matching JSON and SSE totals", async () => {
    for (const [details, cached] of [
      [{ prompt_tokens_details: { cached_tokens: 0 } }, 0],
      [{ prompt_tokens_details: { cached_tokens: 7 } }, 7],
      [{ cached_prompt_tokens: 4 }, 4],
      [{ prompt_tokens_details: { cached_tokens: 2 }, cached_prompt_tokens: 6 }, 2],
      [{ prompt_tokens_details: {}, cached_prompt_tokens: 3 }, 3],
    ] as const) {
      const usage = { prompt_tokens: 7, completion_tokens: 3, ...details };
      const json = await claudeResponse(Response.json({ choices: [{ message: { content: "答" }, finish_reason: "stop" }], usage }), false, mask);
      const counts = (await json.json() as any).usage;
      expect(counts).toEqual({ input_tokens: 7 - cached, output_tokens: 3, cache_read_input_tokens: cached, cache_creation_input_tokens: 0 });
      const source = delta("答") + encode({ choices: [{ delta: {}, finish_reason: "stop" }], usage }) + "data: [DONE]\n\n";
      const events = frames(await (await claudeResponse(new Response(source), true, mask)).text());
      expect(events.find(event => event.type === "message_delta").usage).toEqual(counts);
      expect(counts.input_tokens + counts.cache_read_input_tokens + counts.cache_creation_input_tokens + counts.output_tokens).toBe(10);
      expect(events.at(-1).type).toBe("message_stop");
    }
    // Later cumulative usage replaces earlier usage; cache reads are not summed across events.
    const source = delta("答") + encode({ choices: [], usage: { prompt_tokens: 7, completion_tokens: 1, prompt_tokens_details: { cached_tokens: 2 } } })
      + encode({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 3, prompt_tokens_details: { cached_tokens: 4 } } }) + "data: [DONE]\n\n";
    const events = frames(await (await claudeResponse(new Response(source), true, mask)).text());
    expect(events.find(event => event.type === "message_delta").usage).toEqual({ input_tokens: 3, output_tokens: 3, cache_read_input_tokens: 4, cache_creation_input_tokens: 0 });
  });
  test("rejects invalid cache counters in both JSON and SSE without successful completion", async () => {
    for (const cached of [-1, 8, .5, "2", true, null, {}, [], Number.MAX_SAFE_INTEGER + 1]) {
      for (const details of [{ prompt_tokens_details: { cached_tokens: cached }, cached_prompt_tokens: 0 }, { cached_prompt_tokens: cached }]) {
        const usage = { prompt_tokens: 7, completion_tokens: 3, ...details };
        const json = await claudeResponse(Response.json({ choices: [{ message: { content: "答" }, finish_reason: "stop" }], usage }), false, mask);
        expect(json.status).toBe(502);
        expect(await json.text()).toContain("Invalid model cache usage");
        const source = delta("答") + encode({ choices: [{ delta: {}, finish_reason: "stop" }], usage }) + "data: [DONE]\n\n";
        const events = frames(await (await claudeResponse(new Response(source), true, mask)).text());
        expect(events.at(-1)).toMatchObject({ type: "error", error: { message: "模型返回的缓存用量无效 / Invalid model cache usage" } });
        expect(events.some(event => event.type === "message_delta" || event.type === "message_stop")).toBe(false);
      }
    }
  });
  test("streams UTF-8 split at every byte with CRLF frames and exact final usage", async () => {
    const bytes = new TextEncoder().encode((delta("你好🌱") + final("length") + "data: [DONE]\n\n").replaceAll("\n", "\r\n"));
    let index = 0;
    const source = new ReadableStream<Uint8Array>({ pull(controller) {
      if (index === bytes.length) controller.close(); else controller.enqueue(bytes.slice(index, ++index));
    } });
    const response = await claudeResponse(new Response(source), true, mask);
    const events = frames(await response.text());
    expect(events.find(event => event.type === "content_block_delta").delta.text).toBe("你好🌱");
    expect(events.at(-2)).toMatchObject({ delta: { stop_reason: "max_tokens" }, usage: { input_tokens: 7, output_tokens: 3 } });
    expect(events.at(-1).type).toBe("message_stop");
  });
  test("truncated and failed streams emit errors and never a successful message_stop", async () => {
    for (const text of [delta("partial"), delta("partial") + "data: [DONE]\n\n",
      delta("partial") + encode({ error: { message: "private-key failed" } }), final("cancelled") + "data: [DONE]\n\n",
      final() + delta("after finish") + "data: [DONE]\n\n"]) {
      const response = await claudeResponse(new Response(text), true, mask);
      const events = frames(await response.text());
      expect(events.at(-1).type).toBe("error");
      expect(events.some(event => event.type === "message_stop")).toBe(false);
      expect(JSON.stringify(events)).not.toContain("private-key");
    }
  });
  test("downstream cancellation releases a stalled upstream reader", async () => {
    let cancelled = false;
    const source = new ReadableStream<Uint8Array>({ cancel() { cancelled = true; } });
    const response = await claudeResponse(new Response(source), true, mask);
    const reader = response.body!.getReader();
    await reader.read();
    const pending = reader.read();
    await reader.cancel(); await pending;
    expect(cancelled).toBe(true);
  });
});


const tool = { name: "weather", description: "查天气", input_schema: { type: "object", properties: { city: { type: "string" } }, required: ["city"] } };
const toolPayload = { ...payload, tools: [tool] };
const toolHistory = [
  { role: "user", content: "查北京和上海" },
  { role: "assistant", content: [{ type: "text", text: "我来查询" },
    { type: "tool_use", id: "tool_a", name: "weather", input: { city: "北京" } },
    { type: "tool_use", id: "tool_b", name: "weather", input: { city: "上海" } }] },
  { role: "user", content: [{ type: "tool_result", tool_use_id: "tool_b", content: [{ type: "text", text: "晴" }] },
    { type: "tool_result", tool_use_id: "tool_a", content: "连接失败", is_error: true }, { type: "text", text: "先回答上海" }] },
];
const call = { id: "tool_a", type: "function", function: { name: "weather", arguments: '{"city":"北京"}' } };
describe("Claude tool protocol", () => {
  test("preserves tool results when fewer than 100 source messages expand beyond 100", () => {
    const calls = Array.from({ length: 100 }, (_, index) => ({
      type: "tool_use", id: `tool_${index}`, name: "weather", input: { city: `城市${index}` },
    }));
    const results = calls.map(call => ({ type: "tool_result", tool_use_id: call.id, content: `结果${call.id}` }));
    const converted = claudeRequest({ ...toolPayload, system: "规则", messages: [
      { role: "user", content: "查天气" },
      { role: "assistant", content: calls },
      { role: "user", content: [...results, { type: "text", text: "汇总所有结果" }] },
    ] }, "2023-06-01");
    expect(converted.messages).toEqual([
      { role: "system", content: "规则" }, { role: "user", content: "查天气" },
      { role: "assistant", content: null, tool_calls: calls.map(call => ({
        id: call.id, type: "function", function: { name: call.name, arguments: JSON.stringify(call.input) },
      })) },
      ...results.map(result => ({ role: "tool", tool_call_id: result.tool_use_id, content: result.content })),
      { role: "user", content: "汇总所有结果" },
    ]);
  });
  test("converts parallel tool history without losing IDs, result order, error indication or user follow-up", () => {
    const converted = claudeRequest({ ...toolPayload, messages: toolHistory, tool_choice: { type: "any", disable_parallel_tool_use: true } }, "2023-06-01");
    expect(converted).toMatchObject({ tools: [{ type: "function", function: { name: "weather", description: "查天气", parameters: tool.input_schema } }], tool_choice: "required", parallel_tool_calls: false });
    expect(converted.messages).toEqual([
      { role: "user", content: "查北京和上海" },
      { role: "assistant", content: "我来查询", tool_calls: [call, { id: "tool_b", type: "function", function: { name: "weather", arguments: '{"city":"上海"}' } }] },
      { role: "tool", tool_call_id: "tool_b", content: "晴" },
      { role: "tool", tool_call_id: "tool_a", content: "[Tool error]\n连接失败" },
      { role: "user", content: "先回答上海" },
    ]);
    const onlyCalls = [{ role: "assistant", content: [{ type: "tool_use", id: "a", name: "weather", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "a" }] }];
    expect(claudeRequest({ ...toolPayload, messages: onlyCalls }, "2023-06-01").messages).toEqual([
      { role: "assistant", content: null, tool_calls: [{ id: "a", type: "function", function: { name: "weather", arguments: "{}" } }] },
      { role: "tool", tool_call_id: "a", content: "" },
    ]);
  });
  test("maps named, auto and none choices and rejects unsupported or ambiguous tool inputs", () => {
    expect(claudeRequest({ ...toolPayload, tool_choice: { type: "tool", name: "weather", disable_parallel_tool_use: false } }, "2023-06-01"))
      .toMatchObject({ tool_choice: { type: "function", function: { name: "weather" } }, parallel_tool_calls: true });
    expect(claudeRequest({ ...toolPayload, tool_choice: { type: "auto" } }, "2023-06-01").tool_choice).toBe("auto");
    expect(claudeRequest({ ...payload, tools: [{ ...tool, name: "mcp_" + "x".repeat(100) }] }, "2023-06-01").tools![0].function.name).toHaveLength(104);
    expect(claudeRequest({ ...payload, tool_choice: { type: "none" } }, "2023-06-01").tool_choice).toBe("none");
    for (const body of [
      { ...toolPayload, tools: [tool, tool] }, { ...toolPayload, tools: [{ ...tool, input_schema: [] }] },
      { ...toolPayload, tool_choice: { type: "tool", name: "unknown" } },
      { ...toolPayload, tool_choice: { type: "any", name: "weather" } },
      { ...toolPayload, tool_choice: { type: "any", disable_parallel_tool_use: "yes" } },
      { ...toolPayload, messages: [toolHistory[0], toolHistory[1], { role: "user", content: "forgot results" }] },
      { ...toolPayload, messages: [{ role: "user", content: [{ type: "tool_result", tool_use_id: "unknown", content: "x" }] }] },
      { ...toolPayload, messages: [toolHistory[0], toolHistory[1], { role: "user", content: [{ type: "tool_result", tool_use_id: "tool_a", content: [{ type: "image" }] }] }] },
      { ...toolPayload, messages: [{ role: "assistant", content: [{ type: "tool_use", id: "a", name: "weather", input: {} }, { type: "text", text: "after" }] }, { role: "user", content: "x" }] },
    ]) expect(() => claudeRequest(body, "2023-06-01")).toThrow();
  });
  test("returns validated tool_use blocks and tool_use stop reason, preserving exact JSON numbers", async () => {
    const response = await claudeResponse(Response.json({ choices: [{ message: { content: "查询", tool_calls: [call] }, finish_reason: "tool_calls" }],
      usage: { prompt_tokens: 17, completion_tokens: 8 } }), false, mask);
    expect(await response.json()).toMatchObject({ content: [{ type: "text", text: "查询" },
      { type: "tool_use", id: "tool_a", name: "weather", input: { city: "北京" } }], stop_reason: "tool_use", usage: { input_tokens: 17, output_tokens: 8 } });
    const exact = await claudeResponse(Response.json({ choices: [{ message: { content: null, tool_calls: [{ ...call, function: { name: "weather", arguments: '{"id":9007199254740993,"ratio":1.0}' } }] }, finish_reason: "tool_calls" }],
      usage: { prompt_tokens: 17, completion_tokens: 8 } }), false, mask);
    expect(await exact.text()).toContain('"input":{"id":9007199254740993,"ratio":1.0}');
  });
  test("maps interleaved OpenAI tool argument chunks into sequential Claude JSON blocks with final usage", async () => {
    const chunks = [delta("查询中"),
      encode({ choices: [{ delta: { tool_calls: [{ index: 0, id: "tool_a", type: "function", function: { name: "weather", arguments: '{"city":' } },
        { index: 1, id: "tool_b", type: "function", function: { name: "weather", arguments: '{"city":"上' } }] } }] }),
      encode({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: '"北京"}' } }, { index: 1, function: { arguments: '海"}' } }] } }] }),
      final("tool_calls"), "data: [DONE]\n\n"].join("");
    const events = frames(await (await claudeResponse(new Response(chunks), true, mask)).text());
    expect(events.filter(event => event.type === "content_block_start").map(event => event.content_block)).toEqual([
      { type: "text", text: "" }, { type: "tool_use", id: "tool_a", name: "weather", input: {} }, { type: "tool_use", id: "tool_b", name: "weather", input: {} },
    ]);
    expect(events.filter(event => event.delta?.type === "input_json_delta").map(event => [event.index, event.delta.partial_json])).toEqual([[1, '{"city":"北京"}'], [2, '{"city":"上海"}']]);
    expect(events.at(-2)).toMatchObject({ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { input_tokens: 7, output_tokens: 3 } });
    expect(events.at(-1).type).toBe("message_stop");
  });
  test("never emits executable tool blocks or success on invalid JSON, duplicate IDs or mismatched stop reason", async () => {
    for (const calls of [[{ ...call, function: { name: "weather", arguments: "{" } }], [call, call], [{ ...call, function: { name: "weather", arguments: "[]" } }], []]) {
      const response = await claudeResponse(Response.json({ choices: [{ message: { content: null, tool_calls: calls }, finish_reason: "tool_calls" }], usage: { prompt_tokens: 1, completion_tokens: 1 } }), false, mask);
      expect(response.status).toBe(502);
    }
    for (const [argumentsText, reason] of [["{", "tool_calls"], ["{}", "stop"]]) {
      const source = encode({ choices: [{ delta: { tool_calls: [{ index: 0, ...call, function: { name: "weather", arguments: argumentsText } }] } }] }) + final(reason) + "data: [DONE]\n\n";
      const events = frames(await (await claudeResponse(new Response(source), true, mask)).text());
      expect(events.at(-1).type).toBe("error");
      expect(events.some(event => event.content_block?.type === "tool_use" || event.type === "message_stop")).toBe(false);
    }
  });
});


describe("pi-ai default Anthropic wire hints", () => {
  test("accepts default ephemeral breakpoints and eager tool streaming hints without inventing cache usage", async () => {
    const converted = claudeRequest({ ...toolPayload, stream: true, thinking: { type: "disabled" }, stop_sequences: [],
      system: [{ type: "text", text: "规则", cache_control: { type: "ephemeral", ttl: "1h" } }],
      tools: [{ ...tool, eager_input_streaming: true, cache_control: { type: "ephemeral" } }],
      messages: [{ role: "user", content: [{ type: "text", text: "上海", cache_control: { type: "ephemeral", ttl: "5m" } }] }],
    }, "2023-06-01");
    expect(converted).toMatchObject({ stream: true, thinking: "off", messages: [{ role: "system", content: "规则" }, { role: "user", content: "上海" }] });
    expect(JSON.stringify(converted)).not.toContain("cache_control");
    expect(JSON.stringify(converted)).not.toContain("eager_input_streaming");
    const history = structuredClone(toolHistory);
    (history[2]!.content as any[])[1].cache_control = { type: "ephemeral" };
    expect(claudeRequest({ ...toolPayload, messages: history }, "2023-06-01").messages.at(-1)).toEqual({ role: "user", content: "先回答上海" });
    const response = await claudeResponse(Response.json({ choices: [{ message: { content: "答" }, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 3 } }), false, mask);
    expect((await response.json() as any).usage).toEqual({ input_tokens: 7, output_tokens: 3 });
  });
  test("rejects invalid cache and streaming hints instead of silently treating arbitrary options as compatible", () => {
    for (const cache_control of [null, {}, "ephemeral", { type: "persistent" }, { type: "ephemeral", ttl: "2h" }, { type: "ephemeral", ttl: 300 }, { type: "ephemeral", scope: "global" }]) {
      expect(() => claudeRequest({ ...payload, system: [{ type: "text", text: "x", cache_control }] }, "2023-06-01")).toThrow();
    }
    for (const eager_input_streaming of [null, 1, "true", {}]) {
      expect(() => claudeRequest({ ...toolPayload, tools: [{ ...tool, eager_input_streaming }] }, "2023-06-01")).toThrow();
    }
    expect(claudeRequest({ ...toolPayload, tools: [{ ...tool, eager_input_streaming: false }] }, "2023-06-01").tools).toHaveLength(1);
  });
});
