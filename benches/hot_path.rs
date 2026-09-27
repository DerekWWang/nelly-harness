use nelly_harness::{notes::Notes, schedule::Schedule, Harness, ToolCall};
use std::{hint::black_box, time::Instant};

fn bench(name: &str, iterations: u32, mut run: impl FnMut()) {
    for _ in 0..10_000 {
        run();
    }
    let started = Instant::now();
    for _ in 0..iterations {
        run();
    }
    let ns = started.elapsed().as_nanos() as f64 / f64::from(iterations);
    println!("{name}: {ns:.1} ns/op ({iterations} iterations)");
}

fn main() {
    let mut notes = Notes::default();
    for i in 0..1000 {
        notes
            .put("people".into(), format!("person-{i}"), "likes tea".into())
            .unwrap();
    }
    let mut schedule = Schedule::default();
    for day in 0..30 {
        schedule
            .create(format!("meeting-{day}"), day * 1440 + 540, day * 1440 + 600)
            .unwrap();
    }
    let mut h = Harness::default();
    h.execute(
        &ToolCall::NotesPut {
            topic: "people".into(),
            name: "Ada".into(),
            note: "likes tea".into(),
        },
        false,
    )
    .unwrap();
    let call = ToolCall::NotesGet {
        topic: "people".into(),
        name: "Ada".into(),
    };
    h.execute(&call, true).unwrap();
    bench("borrowed note lookup / 1000 notes", 1_000_000, || {
        black_box(notes.get(black_box("people"), black_box("person-555")));
    });
    bench("bitmap 30-minute availability / 30 days", 1_000_000, || {
        black_box(schedule.is_free(black_box(600), black_box(630)).unwrap());
    });
    bench("bitmap first free hour / 30 days", 100_000, || {
        black_box(
            schedule
                .first_free(black_box(540), black_box(30 * 1440), 60)
                .unwrap(),
        );
    });
    bench("cached typed tool dispatch", 1_000_000, || {
        black_box(h.execute(black_box(&call), false).unwrap());
    });
    let encoded_call = serde_json::to_vec(&call).unwrap();
    bench(
        "JSON parse + cached dispatch + result serialization",
        100_000,
        || {
            let parsed: ToolCall = serde_json::from_slice(black_box(&encoded_call)).unwrap();
            let result = h.execute(&parsed, false).unwrap();
            black_box(serde_json::to_vec(&result).unwrap());
        },
    );
    println!(
        "30 occupied calendar days: {} bitmap payload bytes",
        schedule.bitmap_bytes()
    );
    println!("Excludes I/O transport, disk sync, model inference and Memorable. Only the explicitly labeled JSON benchmark includes serialization.");
}
