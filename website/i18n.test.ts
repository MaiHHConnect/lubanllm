import { expect, test } from "bun:test";
import { assertBilingualCatalog, messages, domainNames, localizedCountry } from "./i18n";

test("website gate rejects missing/empty/mismatched or untranslated English content",()=> {
  expect(()=>assertBilingualCatalog()).not.toThrow();
  const copy=()=>structuredClone(messages) as Record<string,Record<string,string>>;
  let value=copy();delete value.en.profileHistory;expect(()=>assertBilingualCatalog(value)).toThrow("keys must match");
  value=copy();value.en.profileHistory=" ";expect(()=>assertBilingualCatalog(value)).toThrow("missing translation");
  value=copy();value.en.profileNewCorrect="New answers";expect(()=>assertBilingualCatalog(value)).toThrow("mismatched placeholders");
  value=copy();value.en.profileHistory="成长经历";expect(()=>assertBilingualCatalog(value)).toThrow("untranslated English");
  const categories=structuredClone(domainNames) as Record<string,Record<string,string>>;
  delete categories.en.pets;expect(()=>assertBilingualCatalog(messages,categories)).toThrow("keys must match");
});
test("country flags and country names follow the selected language",()=> {
  expect(localizedCountry("US","en")).toBe("🇺🇸 United States");
  expect(localizedCountry("US","zh")).toBe("🇺🇸 美国");
  expect(localizedCountry("invalid","en")).toBe("Not selected");
});
