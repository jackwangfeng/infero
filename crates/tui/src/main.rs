//! tuili-chat — a terminal client for the tuili server.
//!
//! Talks to any OpenAI-compatible `/v1/chat/completions` endpoint over SSE.

use tuili_tui::{app, client, ui};

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{
    self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind,
};

use app::{App, InFlight, Role};

/// How often to look for terminal input. Also the ceiling on how long a
/// freshly arrived token waits before being drawn.
const TICK: Duration = Duration::from_millis(16);
/// The health endpoint is polled far more slowly; it only feeds the header.
const HEALTH_EVERY: Duration = Duration::from_secs(5);

struct Args {
    addr: String,
    model: Option<String>,
    system: Option<String>,
    temperature: f32,
    max_tokens: usize,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        addr: "127.0.0.1:8080".into(),
        model: None,
        system: None,
        temperature: 0.7,
        max_tokens: 512,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--host" | "-H" => {
                args.addr = it.next().context("--host needs an address")?;
            }
            "--model" | "-m" => args.model = it.next(),
            "--system" | "-s" => args.system = it.next(),
            "--temperature" | "-t" => {
                args.temperature = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .context("--temperature needs a number")?;
            }
            "--max-tokens" | "-n" => {
                args.max_tokens = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .context("--max-tokens needs a number")?;
            }
            "--help" | "-h" => {
                println!(
                    "tuili-chat — terminal client for an OpenAI-compatible server\n\n\
                       -H, --host <addr>         default 127.0.0.1:8080\n\
                       -m, --model <name>        default: whatever the server reports\n\
                       -s, --system <prompt>     prepend a system message\n\
                       -t, --temperature <f>     default 0.7\n\
                       -n, --max-tokens <n>      default 512\n"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unexpected argument {other:?} (try --help)"),
        }
    }
    // Accept a URL for the host as a convenience.
    args.addr = args
        .addr
        .trim_start_matches("http://")
        .trim_end_matches('/')
        .to_string();
    if !args.addr.contains(':') {
        args.addr.push_str(":8080");
    }
    Ok(args)
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let health = client::health(&args.addr).with_context(|| {
        format!(
            "cannot reach a tuili server at {}\n\nStart one with:\n  \
             tuili --model <model.gguf> --host {}",
            args.addr, args.addr
        )
    })?;

    let mut app = App::new(args.addr.clone(), health, args.system);
    app.temperature = args.temperature;
    app.max_tokens = args.max_tokens;
    let model = args.model.unwrap_or_else(|| app.health.model.clone());

    let mut terminal = ratatui::init();
    crossterm::execute!(std::io::stdout(), event::EnableMouseCapture).ok();
    let result = run(&mut terminal, &mut app, &model);
    crossterm::execute!(std::io::stdout(), event::DisableMouseCapture).ok();
    ratatui::restore();
    result
}

fn run(terminal: &mut ratatui::DefaultTerminal, app: &mut App, model: &str) -> Result<()> {
    let mut dirty = true;
    let mut last_health = Instant::now();

    loop {
        if dirty {
            terminal.draw(|f| ui::draw(f, app))?;
            dirty = false;
        }

        if event::poll(TICK)? {
            match event::read()? {
                TermEvent::Key(key) if key.kind == KeyEventKind::Press => {
                    handle_key(app, key, model);
                    dirty = true;
                }
                TermEvent::Mouse(m) => match m.kind {
                    MouseEventKind::ScrollUp => {
                        app.scroll_up(3);
                        dirty = true;
                    }
                    MouseEventKind::ScrollDown => {
                        app.scroll_down(3);
                        dirty = true;
                    }
                    _ => {}
                },
                TermEvent::Resize(_, _) => dirty = true,
                _ => {}
            }
        }

        if app.poll() {
            dirty = true;
        }
        if app.should_quit {
            return Ok(());
        }

        // Keep the header's queue depth roughly current without hammering the
        // server; it shares a thread with nothing, so this is a blocking call.
        if !app.is_generating() && last_health.elapsed() > HEALTH_EVERY {
            last_health = Instant::now();
            if let Ok(h) = client::health(&app.addr) {
                app.health = h;
                dirty = true;
            }
        }
    }
}

fn handle_key(app: &mut App, key: KeyEvent, model: &str) {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    match key.code {
        KeyCode::Char('c') if ctrl => {
            // One ctrl+c stops a reply, a second one leaves.
            if app.is_generating() {
                app.cancel();
            } else {
                app.should_quit = true;
            }
        }
        KeyCode::Char('d') if ctrl && app.input.is_empty() => app.should_quit = true,
        KeyCode::Esc if app.is_generating() => app.cancel(),
        KeyCode::Char('l') if ctrl => {
            app.cancel();
            app.messages.retain(|m| m.role == Role::System);
            app.scroll_to_bottom();
        }
        KeyCode::Char('w') if ctrl => app.delete_word(),
        KeyCode::Char('u') if ctrl => {
            app.input.clear();
            app.cursor = 0;
        }
        KeyCode::Char('a') if ctrl => app.move_home(),
        KeyCode::Char('e') if ctrl => app.move_end(),

        KeyCode::Enter if alt || ctrl => app.insert('\n'),
        KeyCode::Enter => send(app, model),

        KeyCode::Backspace => app.backspace(),
        KeyCode::Delete => app.delete(),
        KeyCode::Left => app.move_left(),
        KeyCode::Right => app.move_right(),
        KeyCode::Home => app.move_home(),
        KeyCode::End => app.move_end(),
        KeyCode::PageUp => app.scroll_up(10),
        KeyCode::PageDown => app.scroll_down(10),
        KeyCode::Up if ctrl => app.scroll_up(1),
        KeyCode::Down if ctrl => app.scroll_down(1),

        KeyCode::Char(ch) => app.insert(ch),
        _ => {}
    }
}

fn send(app: &mut App, model: &str) {
    if app.is_generating() {
        app.status = Some("still generating — esc to stop".into());
        return;
    }
    let text = app.input.trim().to_string();
    if text.is_empty() {
        return;
    }
    app.take_input();

    app.push(Role::User, text);
    let request = serde_json::json!({
        "model": model,
        "messages": app.api_messages(),
        "temperature": app.temperature,
        "max_tokens": app.max_tokens,
        "stream": true,
    });
    // The empty assistant turn is what the deltas append to.
    app.push(Role::Assistant, String::new());

    let (tx, rx) = mpsc::channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let addr = app.addr.clone();
    let worker_cancel = cancel.clone();
    std::thread::Builder::new()
        .name("tuili-chat-stream".into())
        .spawn(move || client::stream_chat(&addr, request, worker_cancel, tx))
        .expect("spawning the request thread");

    app.in_flight = Some(InFlight { events: rx, cancel });
}
