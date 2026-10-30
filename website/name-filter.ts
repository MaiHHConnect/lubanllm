import { readFileSync } from "node:fs";
const compact = (text: string) => [...text.normalize("NFKC")].filter(c => c === "*" || !/[\p{P}\p{Z}]/u.test(c)).map(c => c.toLowerCase()).join("");
const words = new Set<string>((JSON.parse(readFileSync(`${import.meta.dir}/name_words.json`,"utf8")) as string[]).map(compact));
const lengths = [...new Set([...words].map(word => [...word].length))].sort((a,b)=>b-a);

export function maskName(text: string): string {
  const chars = [...text.normalize("NFKC")];
  const positions: number[] = [], key: string[] = [];
  chars.forEach((c,i) => {
    if (c === "*" || !/[\p{P}\p{Z}]/u.test(c)) for(const letter of c.toLowerCase()) {positions.push(i); key.push(letter);}
  });
  const output: string[] = [];
  let offset=0, start=0;
  while (offset<key.length) {
    const length=lengths.find(n => offset+n<=key.length && words.has(key.slice(offset,offset+n).join("")));
    if (length) {
      const begin=positions[offset], end=positions[offset+length-1]+1;
      output.push(chars.slice(start,begin).join(""),"***"); start=end; offset+=length;
    } else offset++;
  }
  return (output.join("")+chars.slice(start).join("")).replace(/(?:\*\*\*){2,}/g,"***");
}
