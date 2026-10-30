/** Release-only Mach-O inspection, preserving the existing all-architecture debug gate. */
export function checkReleaseMachO(data: Buffer, name: string) {
  const fail = (message: string): never => { throw Error(message + "：" + name); };
  const need = (ok: unknown, message: string) => { if (!ok) fail(message); };
  function integer(at: number, width: 4 | 8, big: boolean, end = data.length): number {
    need(Number.isSafeInteger(at) && at >= 0 && at <= end - width && end <= data.length, "Mach-O结构截断");
    const value = width === 4 ? (big ? data.readUInt32BE(at) : data.readUInt32LE(at))
      : Number(big ? data.readBigUInt64BE(at) : data.readBigUInt64LE(at));
    need(Number.isSafeInteger(value), "Mach-O整数越界"); return value;
  }
  const string = (at: number) => data.subarray(at, at + 16).toString("ascii").replace(/\0+$/, "");
  function thin(start: number, length: number) {
    const end = start + length, magic = data.subarray(start, start + 4).toString("hex");
    need(Number.isSafeInteger(end) && start >= 0 && end <= data.length && length >= 28, "Mach-O头截断");
    need(["cffaedfe", "feedfacf", "cefaedfe", "feedface"].includes(magic), "发行程序不是有效Mach-O");
    const big = magic.startsWith("feed"), wide = ["cffaedfe", "feedfacf"].includes(magic), header = wide ? 32 : 28;
    need(length >= header, "Mach-O头截断");
    const u32 = (at: number) => integer(at, 4, big, end), commands = u32(start + 16);
    let position = start + header; const limit = position + u32(start + 20);
    need(limit <= end, "Mach-O命令越界");
    for (let commandIndex = 0; commandIndex < commands; commandIndex++) {
      need(position <= limit - 8, "Mach-O命令截断");
      const command = u32(position), size = u32(position + 4);
      need(size >= 8 && size <= limit - position, "Mach-O命令长度无效");
      if (command === 1 || command === 25) {
        const minimum = command === 25 ? 72 : 56, sectionSize = command === 25 ? 80 : 68;
        need(size >= minimum, "Mach-O段截断");
        const sections = u32(position + (command === 25 ? 64 : 48));
        need(sections <= Math.floor((size - minimum) / sectionSize), "Mach-O节越界");
        need(string(position + 8) !== "__DWARF", "发行程序包含DWARF调试信息");
        for (let i = 0; i < sections; i++) need(!/^__z?debug/.test(string(position + minimum + i * sectionSize)), "发行程序包含调试节");
      }
      if (command === 2) {
        need(size >= 24, "Mach-O符号表命令截断");
        const offset = u32(position + 8), count = u32(position + 12), strings = u32(position + 16), stringBytes = u32(position + 20), width = wide ? 16 : 12;
        need(offset <= length && count <= Math.floor((length - offset) / width) && strings <= length && stringBytes <= length - strings, "Mach-O符号表越界");
        for (let i = 0; i < count; i++) {
          const entry = start + offset + i * width, kind = data[entry + 4]!;
          if (!(kind & 0xe0)) continue;
          const index = u32(entry); need(index < stringBytes, "Mach-O符号名称越界");
          const begin = start + strings + index, terminator = data.indexOf(0, begin);
          const description = big ? data.readUInt16BE(entry + 6) : data.readUInt16LE(entry + 6);
          const value = integer(entry + 8, wide ? 8 : 4, big, end);
          const placeholder = i === 0 && kind === 0x3c && data[entry + 5] === 0 && description === 0
            && value === 0x05614542 && terminator >= begin && terminator < start + strings + stringBytes
            && data.subarray(begin, terminator).equals(Buffer.from("radr://5614542"));
          need(placeholder, "发行程序包含STABS调试符号");
        }
      }
      position += size;
    }
    need(position === limit, "Mach-O命令总长度不匹配");
  }
  const magic = data.subarray(0, 4).toString("hex");
  if (["cafebabe", "bebafeca", "cafebabf", "bfbafeca"].includes(magic)) {
    const big = magic.startsWith("cafe"), wide = ["cafebabf", "bfbafeca"].includes(magic), width = wide ? 32 : 20;
    const count = integer(4, 4, big); need(count >= 1 && count <= 32 && 8 + count * width <= data.length, "Mach-O多架构表无效");
    for (let i = 0; i < count; i++) {
      const at = 8 + i * width + 8;
      thin(integer(at, wide ? 8 : 4, big), integer(at + (wide ? 8 : 4), wide ? 8 : 4, big));
    }
  } else thin(0, data.length);
}
