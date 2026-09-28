use nelly_harness::{
    layered::LayeredExecutor,
    layers::{MemoryConfig, MemoryLayers},
    memory::{MemorableCli, MemoryStore, StepInput},
    persistence::{bounded_line, DurableHarness, Line, MAX_REQUEST_BYTES},
    session::ToolExecutor,
    Harness, ToolCall,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    env,
    fs::File,
    io::{self, Read, Write},
    path::PathBuf,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("nelly: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        args.push("help".into());
    }
    let command = args.remove(0);
    match command.as_str() {
        "serve" => serve(args),
        "tools" => {
            write_json(&mut io::stdout().lock(), &tool_definitions())?;
            Ok(())
        }
        "memory" => memory(args),
        "voice" => voice(args),
        "help" | "--help" | "-h" => {
            println!("nelly-harness\n\n  serve [--data DIR | --ephemeral] [--memorable]\n  tools                            Print model function schemas\n  voice MANIFEST WAV [--data DIR] [--episode ID] [--task TEXT] [--memorable]\n  memory begin ID TASK [--data DIR]\n  memory append ID STEP.json [--data DIR]\n  memory finish ID SUMMARY [--data DIR]\n  memory export ID [--data DIR]     Training JSONL on stdout\n  memory sync ID [--data DIR]       Explicit Memorable ingest\n  memory recall QUERY              Memorable recall\n  memory show SLUG                 Memorable show\n  memory chain QUERY               Graph-based workflow composition (JSON)\n  memory list [--all]              Stored workflow revisions (JSON)\n  memory status                    Provider configuration/consent\n  memory layers                    Describe the four connected layers\n\n--memorable enables background memory tools for serve/voice.\n--memorable-bin PATH uses an installed CLI instead of pinned npx.\nDefault data directory: ./data. Calendar times are UTC Unix minutes.\nSee README.md, MODEL.md and MEMORY.md.");
            Ok(())
        }
        _ => Err(format!("unknown command {command:?}; use --help")),
    }
}

fn option(args: &mut Vec<String>, flag: &str) -> Result<Option<String>, String> {
    let Some(pos) = args.iter().position(|a| a == flag) else {
        return Ok(None);
    };
    if pos + 1 >= args.len() {
        return Err(format!("{flag} requires a value"));
    }
    args.remove(pos);
    Ok(Some(args.remove(pos)))
}
fn data_dir(args: &mut Vec<String>) -> Result<PathBuf, String> {
    Ok(PathBuf::from(
        option(args, "--data")?.unwrap_or_else(|| "data".into()),
    ))
}

fn flag(args: &mut Vec<String>, name: &str) -> bool {
    match args.iter().position(|arg| arg == name) {
        Some(index) => {
            args.remove(index);
            true
        }
        None => false,
    }
}

fn memorable_cli(args: &mut Vec<String>) -> Result<MemorableCli, String> {
    let mut cli = MemorableCli::default();
    if let Some(program) = option(args, "--memorable-bin")? {
        cli.program = program.into();
        cli.prefix_args.clear();
    }
    Ok(cli)
}

fn memory_layers(
    args: &mut Vec<String>,
    directory: &std::path::Path,
) -> Result<Option<MemoryLayers>, String> {
    let enabled = flag(args, "--memorable");
    if !enabled && args.iter().any(|arg| arg == "--memorable-bin") {
        return Err("--memorable-bin requires --memorable for serve/voice".into());
    }
    let cli = memorable_cli(args)?;
    if !enabled {
        return Ok(None);
    }
    let store = MemoryStore::open(directory.join("episodes")).map_err(|e| e.to_string())?;
    MemoryLayers::new(store, cli, MemoryConfig::default())
        .map(Some)
        .map_err(|e| e.to_string())
}
fn write_json(out: &mut impl Write, value: &impl serde::Serialize) -> Result<(), String> {
    serde_json::to_writer(&mut *out, value).map_err(|e| e.to_string())?;
    out.write_all(b"\n").map_err(|e| e.to_string())?;
    out.flush().map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    #[serde(default)]
    request_id: Value,
    #[serde(default)]
    speculative: bool,
    call: ToolCall,
}

enum Engine {
    Ephemeral(Harness),
    Durable(DurableHarness),
}
impl ToolExecutor for Engine {
    fn execute(
        &mut self,
        call: &ToolCall,
        speculative: bool,
    ) -> Result<nelly_harness::ToolResult, String> {
        match self {
            Self::Ephemeral(h) => h.execute(call, speculative),
            Self::Durable(h) => h.execute(call, speculative),
        }
    }
    fn revision(&self) -> u64 {
        match self {
            Self::Ephemeral(h) => h.revision(),
            Self::Durable(h) => h.core().revision(),
        }
    }
}

fn serve(mut args: Vec<String>) -> Result<(), String> {
    let ephemeral = if let Some(i) = args.iter().position(|a| a == "--ephemeral") {
        args.remove(i);
        true
    } else {
        false
    };
    if ephemeral && args.iter().any(|a| a == "--data") {
        return Err("choose --ephemeral or --data".into());
    }
    let directory = data_dir(&mut args)?;
    let memory = memory_layers(&mut args, &directory)?;
    if !args.is_empty() {
        return Err("unexpected serve arguments".into());
    }
    let engine = if ephemeral {
        Engine::Ephemeral(Harness::default())
    } else {
        Engine::Durable(DurableHarness::open(directory).map_err(|e| e.to_string())?)
    };
    let mut engine = LayeredExecutor::new(engine, memory);
    let result = (|| -> Result<(), String> {
        let mut input = io::stdin().lock();
        let mut output = io::stdout().lock();
        let mut line = Vec::with_capacity(4096);
        loop {
            let status = bounded_line(&mut input, &mut line, MAX_REQUEST_BYTES)
                .map_err(|e| e.to_string())?;
            if status == Line::Eof {
                break;
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let reply = match serde_json::from_slice::<Request>(&line) {
                Err(error) => json!({"ok":false,"request_id":null,"error":error.to_string()}),
                Ok(request) => match engine.execute(&request.call, request.speculative) {
                    Ok(result) => {
                        json!({"ok":true,"request_id":request.request_id,"result":result.value,"cached":result.cached,"revision":result.revision})
                    }
                    Err(error) => json!({"ok":false,"request_id":request.request_id,"error":error}),
                },
            };
            write_json(&mut output, &reply)?;
            if status == Line::Truncated {
                break;
            }
        }
        Ok(())
    })();
    finish_with_shutdown(result, engine.shutdown())
}

fn finish_with_shutdown(
    result: Result<(), String>,
    shutdown: io::Result<()>,
) -> Result<(), String> {
    match (result, shutdown) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error.to_string()),
        (Err(error), Err(shutdown)) => {
            Err(format!("{error}; memory shutdown also failed: {shutdown}"))
        }
    }
}

fn read_json(path: &str, max: u64) -> Result<Value, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > max {
        return Err("JSON file exceeds limit".into());
    }
    serde_json::from_slice(&bytes).map_err(|e| e.to_string())
}

fn memory(mut args: Vec<String>) -> Result<(), String> {
    let directory = data_dir(&mut args)?;
    let cli = memorable_cli(&mut args)?;
    if args.is_empty() {
        return Err("memory requires a subcommand".into());
    }
    let command = args.remove(0);
    let all = if command == "list" {
        flag(&mut args, "--all")
    } else {
        false
    };
    let required = if matches!(command.as_str(), "begin" | "append" | "finish") {
        2
    } else if matches!(command.as_str(), "list" | "status" | "layers") {
        0
    } else {
        1
    };
    if args.len() != required {
        return Err(format!(
            "memory {command} expects {required} arguments; see --help"
        ));
    }
    if command == "layers" {
        return write_json(
            &mut io::stdout().lock(),
            &json!({"layers":[
            {"layer":1,"name":"traces","implementation":"MemoryStore: durable audio, raw state/action/result episodes"},
            {"layer":2,"name":"workflow_synthesis","implementation":"Memorable ingest: derived trace submitted to the configured provider"},
            {"layer":3,"name":"graph_assembly","implementation":"Memorable chain: ordered workflow composition, dependencies and coverage"},
            {"layer":4,"name":"retrieval","implementation":"background recall/show/list with bounded TTL cache and pending jobs"}
        ],"provider_checked":false}),
        );
    }
    if matches!(
        command.as_str(),
        "recall" | "show" | "chain" | "list" | "status"
    ) {
        let text = match command.as_str() {
            "recall" => cli.recall(&args[0]),
            "show" => cli.show(&args[0]),
            "chain" => cli.chain(&args[0]),
            "list" => cli.list(all),
            _ => cli.status(),
        }
        .map_err(|e| e.to_string())?;
        print!("{}", text.stdout);
        if !text.stderr.is_empty() {
            eprint!("{}", text.stderr);
        }
        return Ok(());
    }
    let store = MemoryStore::open(directory.join("episodes")).map_err(|e| e.to_string())?;
    let mut out = io::stdout().lock();
    match command.as_str() {
        "begin" => write_json(
            &mut out,
            &store.begin(&args[0], &args[1]).map_err(|e| e.to_string())?,
        ),
        "append" => {
            let input: StepInput = serde_json::from_value(read_json(
                &args[1],
                nelly_harness::memory::MAX_STEP_BYTES as u64,
            )?)
            .map_err(|e| e.to_string())?;
            write_json(
                &mut out,
                &store.append(&args[0], input).map_err(|e| e.to_string())?,
            )
        }
        "finish" => write_json(
            &mut out,
            &store
                .finish(&args[0], &args[1])
                .map_err(|e| e.to_string())?,
        ),
        "export" => {
            store
                .export_jsonl(&args[0], &mut out)
                .map_err(|e| e.to_string())?;
            Ok(())
        }
        "sync" => write_json(
            &mut out,
            &store.sync(&args[0], &cli).map_err(|e| e.to_string())?,
        ),
        _ => Err(format!("unknown memory command {command}")),
    }
}

#[cfg(not(feature = "onnx"))]
fn voice(_args: Vec<String>) -> Result<(), String> {
    Err("voice requires cargo build --release --features onnx; see MODEL.md".into())
}

#[cfg(feature = "onnx")]
fn voice(mut args: Vec<String>) -> Result<(), String> {
    use nelly_harness::{
        audio::WavReader,
        memory::{Action, AudioSource},
        model::{AudioChunk, OnnxVoiceModel, VoiceModel},
        session::{SessionEvent, VoiceSession},
    };
    let directory = data_dir(&mut args)?;
    let memory = memory_layers(&mut args, &directory)?;
    let episode = option(&mut args, "--episode")?;
    let task = option(&mut args, "--task")?.unwrap_or_else(|| "Local voice model session".into());
    if args.len() != 2 {
        return Err("voice expects MANIFEST WAV; see --help".into());
    }
    let model = OnnxVoiceModel::load(&args[0])?;
    let mut audio = WavReader::open(&args[1]).map_err(|e| e.to_string())?;
    if audio.sample_rate_hz() != model.sample_rate_hz() {
        return Err("WAV sample rate differs from model; resample before inference".into());
    }
    let engine = DurableHarness::open(&directory).map_err(|e| e.to_string())?;
    let engine = LayeredExecutor::new(engine, memory);
    let sample_rate_hz = model.sample_rate_hz();
    let mut pcm = vec![0.0; model.chunk_samples()];
    let mut session = VoiceSession::new(model, engine);
    let result = (|| -> Result<(), String> {
        let mut audio_refs = Value::Null;
        let store = if let Some(id) = episode.as_ref() {
            let store = MemoryStore::open(directory.join("episodes")).map_err(|e| e.to_string())?;
            store.begin(id, &task).map_err(|e| e.to_string())?;
            // Snapshot audio once before inference, then infer from that owned copy.
            // Even a silent or reply-only session retains its original observation.
            let step=store.append(id,StepInput{
            pre_state:json!({"revision":session.engine().revision(),"model_manifest":args[0]}),
            action:Action{name:"audio_observation".into(),input:json!({"sample_rate_hz":sample_rate_hz,"channels":1})},
            result:Some(json!({"captured":true})),
            post_state:json!({"revision":session.engine().revision()}),started_ms:0,ended_ms:0,
            audio:vec![AudioSource{path:PathBuf::from(&args[1]),role:"observation".into(),media_type:"audio/wav".into(),sample_rate_hz:Some(sample_rate_hz),channels:Some(1)}],
        }).map_err(|e|e.to_string())?;
            audio = WavReader::open(store.root().join(&step.audio[0].path))
                .map_err(|e| e.to_string())?;
            if audio.sample_rate_hz() != sample_rate_hz {
                return Err("audio changed while creating the recording".into());
            }
            audio_refs = serde_json::to_value(step.audio).map_err(|e| e.to_string())?;
            Some(store)
        } else {
            None
        };
        let mut samples = 0u64;
        let mut output = io::stdout().lock();
        let mut last_context = Value::Null;
        while let Some(valid) = audio.read_chunk(&mut pcm).map_err(|e| e.to_string())? {
            let started_ms = samples * 1000 / u64::from(sample_rate_hz);
            samples += valid as u64;
            let ended_ms = samples * 1000 / u64::from(sample_rate_hz);
            let mut revision = session.engine().revision();
            for event in session.process(AudioChunk {
                sample_rate_hz,
                pcm: &pcm,
            })? {
                let mut pre_state = json!({"revision":revision,"last_context":last_context,"audio":audio_refs,"model_manifest":args[0]});
                let (action, result) = match &event {
                    SessionEvent::Reply { text } => (
                        Action {
                            name: "reply".into(),
                            input: json!({"text":text}),
                        },
                        json!({"generated":true}),
                    ),
                    SessionEvent::ToolResult {
                        call,
                        speculative,
                        result,
                        revision_before,
                        revision_after,
                        ..
                    } => {
                        pre_state["revision"] = json!(revision_before);
                        pre_state["speculative"] = json!(speculative);
                        revision = *revision_after;
                        let mut input = call.clone();
                        if let Some(object) = input.as_object_mut() {
                            object.remove("tool");
                        }
                        (
                            Action {
                                name: call["tool"].as_str().unwrap_or("invalid_tool_call").into(),
                                input,
                            },
                            result.clone(),
                        )
                    }
                };
                last_context = serde_json::to_value(&event).map_err(|e| e.to_string())?;
                if let (Some(id), Some(store)) = (episode.as_ref(), store.as_ref()) {
                    store.append(id,StepInput{pre_state,action,result:Some(result),post_state:json!({"revision":revision,"last_context":last_context,"audio":audio_refs}),started_ms,ended_ms,audio:Vec::new()}).map_err(|e|e.to_string())?;
                }
                let mut emitted = last_context.clone();
                emitted["at_ms"] = json!(ended_ms);
                write_json(&mut output, &emitted)?;
            }
        }
        if let (Some(id), Some(store)) = (episode.as_ref(), store.as_ref()) {
            store.finish(id,"Completed local WAV inference; owned input audio, generated replies, and observed tool results recorded.").map_err(|e|e.to_string())?;
        }
        Ok(())
    })();
    let (_, engine) = session.into_parts();
    finish_with_shutdown(result, engine.shutdown())
}

fn tool_definitions() -> Value {
    let s = json!({"type":"string"});
    let n = json!({"type":"integer"});
    let u = json!({"type":"integer","minimum":0});
    let definitions = [
        (
            "notes_get",
            "Get a note by exact topic/name.",
            json!({"topic":s,"name":s}),
            vec!["topic", "name"],
            true,
        ),
        (
            "notes_list",
            "List notes sorted by topic/name; pages max 256.",
            json!({"topic":s,"offset":u,"limit":{"type":"integer","minimum":1,"maximum":256}}),
            vec![],
            true,
        ),
        (
            "notes_put",
            "Create or replace one note (max 16 KiB).",
            json!({"topic":s,"name":s,"note":s}),
            vec!["topic", "name", "note"],
            false,
        ),
        (
            "notes_delete",
            "Delete one note.",
            json!({"topic":s,"name":s}),
            vec!["topic", "name"],
            false,
        ),
        (
            "schedule_get",
            "Get an event by id.",
            json!({"id":u}),
            vec!["id"],
            true,
        ),
        (
            "schedule_list",
            "List overlapping events in a half-open UTC Unix minute interval.",
            json!({"start_minute":n,"end_minute":n,"offset":u,"limit":{"type":"integer","minimum":1,"maximum":256}}),
            vec!["start_minute", "end_minute"],
            true,
        ),
        (
            "schedule_is_free",
            "Test availability using minute bitmaps.",
            json!({"start_minute":n,"end_minute":n}),
            vec!["start_minute", "end_minute"],
            true,
        ),
        (
            "schedule_first_free",
            "Find earliest contiguous free minutes in a window.",
            json!({"start_minute":n,"end_minute":n,"duration_minutes":{"type":"integer","minimum":1}}),
            vec!["start_minute", "end_minute", "duration_minutes"],
            true,
        ),
        (
            "schedule_create",
            "Create an event; overlapping events are allowed.",
            json!({"title":s,"start_minute":n,"end_minute":n}),
            vec!["title", "start_minute", "end_minute"],
            false,
        ),
        (
            "schedule_update",
            "Replace an event by id.",
            json!({"id":u,"title":s,"start_minute":n,"end_minute":n}),
            vec!["id", "title", "start_minute", "end_minute"],
            false,
        ),
        (
            "schedule_delete",
            "Delete an event by id.",
            json!({"id":u}),
            vec!["id"],
            false,
        ),
        (
            "stats",
            "Inspect current state and cache payload counts.",
            json!({}),
            vec![],
            true,
        ),
        ("memorable_recall","Prefetch procedural memory; returns a pending job or a cached result. Poll pending jobs.",json!({"query":s}),vec!["query"],true),
        ("memorable_show","Retrieve a procedure as reference data; returns a job or cached result.",json!({"slug":s}),vec!["slug"],true),
        ("memorable_chain","Compose stored workflows through Memorable's dependency graph; returns a job or cached plan.",json!({"query":s}),vec!["query"],true),
        ("memorable_list","List stored workflows and revisions; returns a job or cached inventory.",json!({"all":{"type":"boolean"}}),vec![],true),
        ("memorable_status","Read provider status in the background; status is not cached.",json!({}),vec![],true),
        ("memorable_poll","Read a memory job's pending, ready or failed result without waiting.",json!({"job_id":u}),vec!["job_id"],true),
        ("memorable_ingest","Submit a finished local episode for workflow synthesis. Requires a committed action.",json!({"episode_id":s}),vec!["episode_id"],false),
        ("memorable_invalidate","Clear memory retrieval cache after external provider changes.",json!({}),vec![],false),
        ("memorable_disable","Revoke local memory access and clear cached results. Re-enable through the operator, not the model.",json!({}),vec![],false),
    ];
    json!(definitions.into_iter().map(|(name,description,properties,required,read_only)|json!({"type":"function","function":{"name":name,"description":description,"parameters":{"type":"object","properties":properties,"required":required,"additionalProperties":false}},"read_only":read_only})).collect::<Vec<_>>())
}
