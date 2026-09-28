# Memorable integration contract

Verified against the published `memorable-cli@0.5.30` package and official docs on
2026-09-27. Package inspection and fake-process tests do not establish that a live
account will admit a particular trace. No account was enabled or user trace sent.

Memorable names four layers: **Traces → Workflow Synthesis → Graph Assembly →
Retrieval**. Nelly retains original audio/state/action episodes locally; the CLI
delegates synthesis and provides procedure retrieval and composition.
[Architecture](https://www.memorable.sh/)

## Commands and returned data

Use `npx --yes --package memorable-cli@0.5.30 memorable <command>`.

| Command | Output/behavior |
| --- | --- |
| `ingest <trace.json>` | Extracts and stores an admitted procedure; text output. Requires credentials and write consent. |
| `recall <query> [--single\|--chain]` | Ranked text or a composed plan; automatic mode is the default. |
| `show <slug>` | Procedure rendered as guarded reference text. |
| `chain <query> --json` | Structured plan; `--render` instead returns guarded text. |
| `list --json [--all]` | Revision groups. JSON includes every revision even without `--all` in this version. |
| `status` | Human-readable backend, consent, service, and queue information; no JSON flag. |

Chain output has `segments: string[]`, `coverage: number`, `score: number`, and
`items`. A procedure item has `kind: "procedure"`, `slug`, `title`, `needsFrom`,
`reason` (`matched`, `bridge`, or `follow-on`), optional `segment`, nullable
`verify`, `writes`, `reads`, nullable `evidence: {ok,total}`, and numeric `level`.
A gap has `kind: "gap"` and `segment`. An empty store can return empty `items`
without explicit gaps. Dependencies derive from artifact paths in the current
working directory; there is no raw graph-export command. List output is an array
of `{intent, preferred, revisions}`; revisions carry `slug`, `revision`, `title`,
`verified`, nullable `stale`, `recalled`, `ok`, `fail`, and optional `last_used_at`.

There is no `explain` CLI command. The read-only `mcp` server provides
`memorable_explain_recall`; its tool results contain text. `skills` exports files
and is not a workflow inspection command.
[CLI reference](https://www.memorable.sh/docs/cli)

## Trace adaptation and the custom-tool limitation

Ingest accepts `session_id`, optional `workflow_id` and `prompt`,
`task_description`, `harness`, and `tool_calls: [{name,input,result?}]`.
The pinned CLI preserves only string input fields `command`, `cmd`, `file_path`,
`filePath`, `path`, `notebook_path`, `pattern`, `url`, `query`, `description`,
`shell_id`, and `bash_id`. Each is capped at 4,000 characters and scrubbed. Results
retain only boolean `ok` and finite numeric `exit_code`. Additional step fields
such as `kind` or `activity_class`, and top-level custom registries, are not
forwarded by ingest. There is no documented registry-registration API.
[Published package](https://registry.npmjs.org/memorable-cli/-/memorable-cli-0.5.30.tgz)

A faithful voice adapter preserves the actual tool name, emits a concise
`description` containing action and non-content identifiers, and includes `ok`
only when the real tool outcome is known. It must not copy note bodies, audio,
conversation text, or serialized full arguments into an allowed field. Failed
calls remain failed; speculative operations are not recorded as committed writes.
[Egress and result contract](https://www.memorable.sh/docs/integrate)

**This preserves metadata, but cannot guarantee synthesis.** The pinned package's
embedded agent guidance says unknown harnesses classify non-shell tools as
`other`; the admission check refuses a draft with no `write` or `execute` step.
The public API documents command-shaped input inference for generic harnesses,
not classification overrides. Renaming a Nelly operation to `Write`/`Edit` while
keeping `harness: "nelly-rust"` has no documented guarantee of write
classification. Claiming another harness or inventing a shell command would
misrepresent execution. Custom voice-only traces may therefore be declined until
Memorable supports their activity schema; the local archive remains complete.
[Extraction API](https://www.memorable.sh/docs/api)

## Consent and configuration boundaries

The status table labels consent `write consent`, with exact values `read-write`,
`read-only`, `deny`, or `unset`. Only `read-write` permits ingest; retrieval should
also stop on `deny` or `unset`. In the pinned implementation, `chain`, `show`,
`list`, MCP retrieval, and recall's automatic chain branch do not consistently
check consent. A caller must gate these paths and its own cached context.

There is no public `config get` API. For local-backend diagnostics only, the pinned
implementation reads `${MEMORABLE_HOME || homedir}/.memorable/config.json` and
uses its `consent` field; malformed/missing/unknown values become `unset`.
`MEMORABLE_BACKEND` (`local`, `gbrain`, `qm`) overrides `config.backend`.
Database-backed consent is stored separately, so reading the local file cannot
authorize those backends. Do not read or rewrite the credential fields.
The MCP status tool returns text lines including `write_consent: <mode>`;
neither it nor ordinary status is a documented structured configuration accessor.
These pinned-source observations are compatibility details, not permission to
bypass the CLI's documented consent policy.
