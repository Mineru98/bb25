/**
 * Tokenizer.
 *
 * Behavior:
 *  - ASCII-only. A token is a maximal run of `[A-Za-z0-9]`.
 *  - `A-Z` is lowercased (ASCII lowercase only); every other character —
 *    whitespace, punctuation, and ANY non-ASCII char (é, 한글, full-width …) —
 *    acts as a delimiter and is dropped.
 *  - No stopword removal, no stemming.
 *
 * IMPORTANT: JS `String.prototype.toLowerCase()`, `\w`, and `\p{...}` are
 * Unicode-aware and would change the intended behavior. We therefore inspect raw
 * char codes and lowercase only `A-Z`.
 */
export class Tokenizer {
  tokenize(text: string): string[] {
    const tokens: string[] = [];
    let current = "";

    for (let i = 0; i < text.length; i++) {
      const code = text.charCodeAt(i);
      const isDigit = code >= 48 && code <= 57; // 0-9
      const isUpper = code >= 65 && code <= 90; // A-Z
      const isLower = code >= 97 && code <= 122; // a-z

      if (isDigit || isLower) {
        current += text[i];
      } else if (isUpper) {
        current += String.fromCharCode(code + 32); // ASCII to-lower
      } else if (current.length > 0) {
        tokens.push(current);
        current = "";
      }
    }

    if (current.length > 0) {
      tokens.push(current);
    }

    return tokens;
  }
}
