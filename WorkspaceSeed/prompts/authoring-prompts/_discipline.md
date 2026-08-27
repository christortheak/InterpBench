## The discipline

You are writing MEASUREMENT INSTRUMENT DATA, not prose. The rows below will be
fed to a model and the difference between them will be recorded as evidence
about that model. Anything that distinguishes the rows other than the thing
being measured becomes part of the measurement — silently, and permanently,
because the file is hashed and pinned before any behaviour is recorded.

Five rules govern every kind of data in this workspace. Each one exists because
a delivery failed on it.

**1. Any surface-computable feature must appear at matched rates.** A word, a
stem, a syntactic frame, a discourse marker, a sentence opener, a punctuation
rate, a length distribution — anything a reader can count WITHOUT understanding
the sentence — must be balanced across the groups being contrasted, and within
each matched pair where the shape has pairs. Assume a reviewer will run
frequency analysis over your whole output. The thing you are trying to express
must be the only signal that survives it.

**2. Symmetrize by ADDITION, never by subtraction.** When one side of a
contrast naturally reaches for a construction the other does not, the repair is
to give the other side its own honest instances of that construction — not to
strip the first side of something constitutive. Stripping flattens the thing
you are measuring; adding balances the surface while leaving it intact. The
same applies to vocabulary: each side's own conceptual vocabulary is
constitutive and must NOT be suppressed — it must simply not leak into the
other side's rows.

**3. Vary form ACROSS rows, hold it constant WITHIN a pair.** This is the rule
deliveries get backwards most often. Two paired rows must share their
syntactic frame, their mood, their tense, their person, and their opening
construction; it is the CONTENT that differs. Variety belongs across pairs, not
between the members of one. A pair whose members differ in grammatical mood has
encoded the thing being measured in the grammar, where it can be read off
without reading the sentence.

**4. Cap the distribution, not a list of words.** No content stem may appear in
more than {{stemCapPercent}}% of the rows of any one file, and no syntactic
frame in more than {{frameCapPercent}}% of them. This is a property of your
whole output, checked by computing the top-20 content tokens per file — it is
not a banned-word list you can satisfy by substitution. A conviction, a stance,
or a capability is not a keyword.

**5. Compute every audit number. Never assert one.** The audit battery at the
end of this prompt is a set of NUMBERS to calculate over your finished output
and report alongside it. "Balanced" and "varied" are not audit results. If a
number fails its threshold, rewrite and report the before/after counts. An
audit you asserted rather than computed is the single most common reason a
delivery is rejected.

Write each row as a fresh act of composition. If you notice yourself reusing a
sentence skeleton from an earlier row, rewrite one of them.
