## Task Requirements
1. Analyze the relationship between the [Web Content] and [Index Data] below
2. When answering the [User Question]:
   - Maintain concise and professional responses
   - Use `⟨number⟩` to cite sources
   - Each claim/statement can reference up to 2 related indices
   - Prioritize citing more specific pages (e.g. ⟨5⟩ is more specific than ⟨1⟩)
3. Output must include:
   - Answer in paragraphs
   - References ordered by citation sequence

## Question
{{question}}

---
## Input Data

### [Web Content]
{{contents}}

### References
{{references}}

---
## Output Format

### Answer:
[Organize response in natural language, add ⟨number⟩ markers after each factual claim, separate paragraphs with blank lines]

### References:
[Renumber according to citation order, use this format (omit if none cited):  
- ⟨Original ID⟩ [Page Title](URL)  
Do not include uncited links]