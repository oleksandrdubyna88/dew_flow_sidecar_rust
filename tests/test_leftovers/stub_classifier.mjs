// A stand-in for the shipped classifier whose answer is WRONG on purpose: it calls a line broad that the file does
// not have. `script_fixtures.rs` points the guard at it to prove an answer that does not hold is one unjudged
// finding, never "nothing found". It holds no prefix and creates nothing.
export function classifyLines(text) {
  return Array.from({ length: text.split("\n").length + 5 }, () => "broad");
}
