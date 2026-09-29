# Parsing Japanese Bibliographic References with LLMs and GROBID

This repository contains the experiment code for comparing large language models (LLMs) and
GROBID on parsing Japanese bibliographic reference strings into BibTeX. It covers running the
parsers and evaluating their outputs against gold records.

The dataset (gold BibTeX records and their renderings in seven citation styles) is distributed
separately at
[kalclab/ja-citation-parsing-dataset](https://github.com/kalclab/ja-citation-parsing-dataset).

## Repository layout

| Path | Contents |
|------|----------|
| `scripts/` | Experiment runner, parsers, prompt, and evaluator |
| `configs/` | The run configurations of the reported experiments (30 runs) |
| `run.sh`, `eval.sh` | Command-line entry points |

## Setup

The code requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.sample .env   # then fill in the API key
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key of the OpenAI-compatible endpoint |
| `LLM_BASE_URL` | Base URL of the endpoint; set it to `https://openrouter.ai/api/v1` to use OpenRouter, as in the reported experiments |

### Dataset

The dataset repository distributes a manifest rather than the data, so the dataset has to be
rebuilt locally first; follow the instructions in its README. Generate the reference strings of
the CSL styles with pandoc 3.10: other versions of pandoc ship different Japanese locale terms
and render some strings differently (e.g. 訳 as 翻訳者).

The rebuild writes the reference strings of the BibTeX and CSL styles to two files, and names
the modified jecon style `jecon-mod`. This code reads a single file and names the style
`jecon-kalc`. The following commands convert the rebuilt dataset into the layout used here:

```bash
DATASET=path/to/ja-citation-parsing-dataset
mkdir -p data/v0.2-20260717
cat "$DATASET"/dataset/ref_strings_bst.jsonl "$DATASET"/dataset/ref_strings_csl.jsonl \
  | sed 's/"style": "jecon-mod"/"style": "jecon-kalc"/' \
  > data/v0.2-20260717/ref_strings.jsonl
cp "$DATASET"/dataset/ja_bib_full.bib "$DATASET"/dataset/ja_bib_derived.bib \
  "$DATASET"/dataset/metadata.csv "$DATASET"/sample_100.txt data/v0.2-20260717/
```

`data/v0.2-20260717/` is the default path of both the runner and the evaluator. The code reads
the following files:

| File | Used by | Contents |
|------|---------|----------|
| `ref_strings.jsonl` | runner, evaluator | Reference strings rendered in each citation style |
| `metadata.csv` | runner, evaluator | Entry type and other metadata of each record |
| `sample_100.txt` | runner, evaluator | Keys of the 100 evaluated records |
| `ja_bib_derived.bib` | evaluator | Gold records (the gold source of the reported results) |
| `ja_bib_full.bib` | evaluator | Complete gold records |

Use `dataset_path` and `sample_path` in a configuration, or `--dataset-path` and
`--sample-path` for the evaluator, to read the data from another location.

The citation styles have different names in the paper, the dataset repository, and this code:

| Paper | Dataset repository | This code |
|-------|--------------------|-----------|
| `jecon` | `jecon-mod` | `jecon-kalc` |
| `ipsjsort` | `ipsjsort` | `ipsjsort` |
| `jsai` | `jsai` | `jsai` |
| `jplain` | `jplain` | `jplain` |
| `sist02` | `sist02` | `sist02` |
| `jpa2022` | `jpa2022` | `jpa2022` |
| `chicago` | `chicago-author-date` | `chicago-author-date` |

### GROBID

The GROBID baseline expects a GROBID server at `http://localhost:8070`. The reported results
were obtained with the `grobid/grobid:0.9.0-full` Docker image:

```bash
docker run --rm --init --ulimit core=0 -p 8070:8070 grobid/grobid:0.9.0-full
```

See the [GROBID documentation](https://grobid.readthedocs.io/) for GPU and memory options. Set
`grobid_base_url` in a configuration to use another server.

## Running the experiments

Each JSON file in `configs/` defines one run. The files at the top level of `configs/` are the
first runs, which process the dataset in a fixed order; `configs/002/` to `configs/005/` hold the
second to fifth runs, which shuffle the order with a seed shared by all LLMs.

```bash
./run.sh configs/llm-sol.json                                        # one run
for config in configs/*.json configs/*/*.json; do ./run.sh "$config"; done   # all 30 runs
```

Each run writes its output to `artifacts/<run_id>/`:

| File | Contents |
|------|----------|
| `manifest.json` | Configuration, code revision, and hashes of the dataset and input order |
| `calls.jsonl` | One row per API or GROBID call, including retries |
| `items.jsonl` | One row per reference with the parsed BibTeX fields |
| `raw/` | Raw requests and responses |

An LLM run sends all references of a citation style in one call (`batch_size: 100`). A call
whose output cannot be parsed is retried with the same prompt, up to 20 attempts. GROBID
receives one reference per call, with consolidation disabled.

### Prompt

All LLMs receive the same system prompt and user prompt, defined in `scripts/prompts.py`. The
system prompt is:

```text
You are a bibliographic reference parser. Convert the provided reference strings into valid BibTeX entries.

For each reference, create a single valid BibTeX entry. Return ONLY the BibTeX entries, separated by blank lines.

Rules:
- Use ONLY values that appear in the reference string. Do not invent or guess.
- Use the reference's input_id as the BibTeX citation key.
- Infer the entry type (article, inproceedings, book, incollection, misc) from the reference content.
- For personal authors, use "family, given" format separated by " and " (e.g., "Yamada, Taro and Suzuki, Hanako"). Preserve the detail level (full names or initials) as found in the reference.
- For institutional authors (organizations, committees, agencies), keep the full name as one author element (e.g., author = {World Health Organization}).

- Do not add any explanatory text outside the BibTeX entries.
```

The user prompt template is:

```text
Convert each reference string to a BibTeX entry. Use the input_id as the citation key.

## References

{references}
```

`{references}` is replaced by the references of the call, one per line, each preceded by its
opaque input identifier:

```text
[ref_0001] 中澤達哉 (2016) 「2015年の歴史学界 回顧と展望――近代ロシア・東欧・北欧」，『史学雑誌』，第125巻，第5号，358–362頁．
[ref_0002] ...
```

The identifiers are numbered consecutively across the styles in the lexicographic order of the
dataset keys, before any shuffling, so a shuffled run sends them out of numerical order. The
dataset key, the citation style, and the record metadata are not sent to the model.

### Configuration fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `run_id` | `str` | yes | Run identifier; the output goes to `artifacts/<run_id>/` |
| `parser` | `"llm"` or `"grobid"` | yes | Parser |
| `model` | `str` | LLM only | Model identifier at the endpoint |
| `batch_size` | `int` | LLM only | References per call, grouped by citation style |
| `order_mode` | `"fixed"` or `"shuffle"` | no | Input order (default: `"fixed"`) |
| `shuffle_seed` | `int` | no | Seed of the shuffle when `order_mode` is `"shuffle"` |
| `temperature` | `float` | no | Sampling temperature |
| `seed` | `int` | no | Seed passed to the provider; providers do not guarantee deterministic output |
| `grobid_base_url` | `str` | no | GROBID server (default: `http://localhost:8070`) |
| `grobid_consolidate` | `int` | no | GROBID consolidation level (default: `0`) |
| `dataset_path` | `str` | no | Dataset directory |
| `sample_path` | `str` | no | File of the evaluated record keys |
| `styles` | `list[str]` | no | Citation styles to process |
| `artifacts_dir` | `str` | no | Output root (default: `artifacts`) |

`scripts.run_experiment` refuses to start from a working tree with uncommitted changes, so that
the recorded code revision identifies the code; `run.sh` passes `--allow-dirty` to lift this
check.

## Evaluation

The evaluator reads the saved `manifest.json` and `items.jsonl` of a run, without calling a
parser again, and writes its results to a separate directory:

```bash
./eval.sh artifacts/llm-sol-01    # writes evaluations/llm-sol-01/
```

`eval.sh` runs the following command, which can also be called directly with other options (see
`uv run python -m scripts.evaluate --help`):

```bash
uv run python -m scripts.evaluate \
  --run-dir artifacts/llm-sol-01 \
  --field-policy observable-fields-v1 \
  --gold-source derived \
  --output-dir evaluations/llm-sol-01
```

The evaluator does not overwrite an existing output directory. Its output consists of
`manifest.json` (input hashes, field policy, and code hash), `items.jsonl` (field-level
results), `summary.jsonl`, `result.json`, and `audit.jsonl`.

### Field policy

The field policy (`observable-fields-v1`) is defined in `scripts/evaluation_field_policy.py`.

#### Eligible fields

A field of an item is scored only when it is *eligible*, that is, when all of the following
hold:

1. The citation style renders the field for the entry type of the item. This scope is given by
   `ALWAYS_FIELD_SCOPES` and shown in the table below.
2. The gold record has a value for the field.
3. The gold value can be located in the reference string after normalisation (Unicode NFKC,
   case folding, and removal of whitespace, quotation marks, and BibTeX grouping braces; for
   persons, the expected truncation marker must also be present or absent). Of the 5,155
   in-scope slots with a gold value in the test set, one fails this check and is excluded.

The table lists, for each field and citation style, the entry types in scope: a for
`article`, i for `inproceedings`, b for `book`, and m for `misc`. A dash marks a field that the
style never renders. A code in parentheses marks a combination that is in scope but for which
no gold record of the 100-record sample has a value, so it yields no eligible slot.

| Field | `jecon-kalc` | `ipsjsort` | `jsai` | `jplain` | `sist02` | `jpa2022` | `chicago-author-date` |
|---|---|---|---|---|---|---|---|
| `author` | aibm | aibm | aibm | aibm | aibm | aibm | aibm |
| `editor` | – | – | – | – | – | – | (b) |
| `translator` | b | – | – | – | b | – | b |
| `title` | aibm | aibm | aibm | aibm | aibm | aibm | aibm |
| `journal` | a | a | a | a | a | a | a |
| `booktitle` | i | i | i | i | i | i | i |
| `series` | b | – | b | b | b | b | b |
| `publisher` | ib | ib | ib | ib | ib | b | b |
| `address` | b | b | b | b | b | – | – |
| `edition` | b | b | b | b | b | – | b |
| `year` | aibm | aibm | aibm | aibm | aibm | aibm | aibm |
| `volume` | ai(b) | ai | ai(b) | ai(b) | a | ai | ai(b) |
| `number` | ai | ai | a | a | a | a | a |
| `pages` | ai | ai | ai | ai | ai | ai | ai |
| `doi` | ai | ai | – | – | – | a | ai |
| `isbn` | – | – | – | – | b | – | – |
| `url` | im | im | m | m | aim | im | im |
| `urldate` | – | m | m | m | m | m | – |
| `organization` | i | i | i | i | i | – | – |

`jecon-kalc` is the modified `jecon` style of the dataset (named `jecon-mod` there), which always
prints the proceedings title of an `inproceedings` entry and prints a repeated author in full.

#### Accepted forms

A prediction matches when it is identical to one of the accepted forms of the gold value. Both
sides are compared after removing balanced BibTeX grouping braces and the backslashes of the
BibTeX escapes `\%`, `\&`, `\_`, `\#`, and `\$`; every other difference between the strings
counts. The accepted forms are the gold value itself and the following alternatives, most of
which are admitted only when the reference string prints them:

| Field | Additional accepted forms |
|---|---|
| `title`, `booktitle`, `journal`, `series`, `publisher`, `organization`, `address` | The form printed in the reference string when it differs from the gold value only in letter case or in the width of spaces (e.g. an initialism that `jplain` prints with only its first letter capital) |
| `edition` | The edition as the style renders it, when printed: 第2版 (`jecon-kalc`), 2 edition (`ipsjsort`), 2 版 (`chicago-author-date`) |
| `pages` | The page range as printed in the reference string, including the closing number shortened in the Chicago manner (e.g. 1234–56) and the separator and its spacing as printed; the gold value and each printed form are also accepted with a hyphen, an en dash, or a double hyphen as the separator |
| `urldate` | The date in the format printed in the reference string: `YYYY-MM-DD`, `YYYY/MM/DD`, `YYYY.MM.DD`, or `YYYY年MM月DD日`, with or without zero padding |
| `author`, `editor`, `translator` | See below |

Person fields are compared as the sequence of persons in gold order, with the letters and digits
of each name concatenated, so punctuation and spaces between names are ignored. For each person,
the accepted forms are the name in either order of family and given names and, for Latin names,
with all given names or only the first given name reduced to initials. A name recorded without
a comma, such as an institutional author, must match as a whole, optionally without the word
"and". When a style truncates the author list (`chicago-author-date` from seven
authors and `jecon-kalc` from nine authors, both keeping the first three), only the printed
persons are expected, followed by a marker of the omission (ほか, a final 他, et al., or
`others`). The combination of forms closest to the prediction is chosen, with ties broken by the
order of the forms above.

<!--
## Citation

If you use this code, please cite:

```bibtex
@inproceedings{seimiya2026icadl,
  author    = {SEIMIYA, Ryotaro},
  title     = {Evaluation of Large Language Models for Japanese Bibliographic Reference Parsing across Diverse Citation Styles},
  booktitle = {Proceedings of the International Conference on Asia-Pacific Digital Libraries (ICADL 2026)},
  year      = {2026},
  pages     = {TODO},
  doi       = {TODO}
}
```
-->

## Licence

The code is released under the MIT Licence; see `LICENSE`.
