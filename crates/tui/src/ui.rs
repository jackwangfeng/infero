//! Rendering.

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::app::{App, Role};
use crate::wrap::{width, wrap};

const USER: Color = Color::Cyan;
const ASSISTANT: Color = Color::Green;
const SYSTEM: Color = Color::Magenta;
const ERROR: Color = Color::Red;
const DIM: Color = Color::DarkGray;

fn role_color(role: Role) -> Color {
    match role {
        Role::User => USER,
        Role::Assistant => ASSISTANT,
        Role::System => SYSTEM,
        Role::Error => ERROR,
    }
}

pub fn draw(f: &mut Frame, app: &mut App) {
    let input_height = input_height(app, f.area().width);
    let areas = Layout::vertical([
        Constraint::Length(1),
        Constraint::Min(3),
        Constraint::Length(input_height),
        Constraint::Length(1),
    ])
    .split(f.area());

    draw_header(f, areas[0], app);
    draw_transcript(f, areas[1], app);
    draw_input(f, areas[2], app);
    draw_footer(f, areas[3], app);
}

fn draw_header(f: &mut Frame, area: Rect, app: &App) {
    let h = &app.health;
    let mut spans = vec![
        Span::styled(" tuili ", Style::default().fg(Color::Black).bg(ASSISTANT)),
        Span::raw(" "),
        Span::styled(&h.model, Style::default().add_modifier(Modifier::BOLD)),
    ];
    let mut facts = vec![h.quantization.clone()];
    if h.kv_quant != "f16" {
        facts.push(format!("kv {}", h.kv_quant));
    }
    if h.offloaded_layers > 0 {
        facts.push(format!("{} layers offloaded", h.offloaded_layers));
    }
    facts.push(format!("{} ctx", h.max_seq));
    if h.max_seqs > 1 {
        facts.push(format!("{} slots", h.max_seqs));
    }
    spans.push(Span::styled(
        format!("  {}", facts.join(" · ")),
        Style::default().fg(DIM),
    ));

    if h.queue_depth > 1 {
        spans.push(Span::styled(
            format!("  {} in flight", h.queue_depth),
            Style::default().fg(Color::Yellow),
        ));
    }

    let right = if app.is_generating() {
        Span::styled(
            " generating ",
            Style::default().fg(Color::Black).bg(Color::Yellow),
        )
    } else {
        Span::styled(format!(" {} ", app.addr), Style::default().fg(DIM))
    };

    let left = Line::from(spans);
    let used = width(&left.to_string());
    let pad = (area.width as usize).saturating_sub(used + width(&right.to_string()));
    let mut all = left.spans;
    all.push(Span::raw(" ".repeat(pad)));
    all.push(right);
    f.render_widget(Paragraph::new(Line::from(all)), area);
}

/// Lay the conversation out as flat lines so scrolling has an exact height.
fn transcript_lines(app: &App, width_cells: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let body_width = width_cells.saturating_sub(2).max(8);

    for (i, msg) in app.messages.iter().enumerate() {
        if i > 0 {
            lines.push(Line::raw(""));
        }
        let color = role_color(msg.role);
        let mut header = vec![Span::styled(
            msg.role.label().to_string(),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )];
        if let Some(stats) = &msg.stats {
            header.push(Span::styled(format!("  {stats}"), Style::default().fg(DIM)));
        }
        lines.push(Line::from(header));

        for wrapped in wrap(&msg.text, body_width) {
            let style = match msg.role {
                Role::Error => Style::default().fg(ERROR),
                Role::System => Style::default().fg(DIM),
                _ => Style::default(),
            };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(wrapped, style),
            ]));
        }
    }
    lines
}

fn draw_transcript(f: &mut Frame, area: Rect, app: &mut App) {
    let inner = Rect {
        x: area.x + 1,
        y: area.y,
        width: area.width.saturating_sub(2),
        height: area.height,
    };
    let lines = transcript_lines(app, inner.width as usize);

    app.last_transcript_height = lines.len();
    app.last_viewport_height = inner.height as usize;

    if lines.is_empty() {
        let hint = Paragraph::new(Line::styled(
            "type a message and press enter",
            Style::default().fg(DIM),
        ));
        f.render_widget(hint, inner);
        return;
    }

    // `scroll` counts lines up from the bottom, so the newest output stays put
    // as it grows rather than sliding away.
    let overflow = lines.len().saturating_sub(inner.height as usize);
    let offset = overflow.saturating_sub(app.scroll.min(overflow));
    f.render_widget(Paragraph::new(lines).scroll((offset as u16, 0)), inner);

    if app.scroll > 0 {
        let marker = format!(" {} lines below ", app.scroll);
        let x = inner.x + inner.width.saturating_sub(width(&marker) as u16 + 1);
        let y = inner.y + inner.height.saturating_sub(1);
        f.render_widget(
            Paragraph::new(Line::styled(
                marker,
                Style::default().fg(Color::Black).bg(Color::Yellow),
            )),
            Rect {
                x,
                y,
                width: inner.width.min(30),
                height: 1,
            },
        );
    }
}

fn input_height(app: &App, total_width: u16) -> u16 {
    let inner = (total_width as usize).saturating_sub(4).max(8);
    let lines = wrap(&app.input, inner).len().max(1);
    (lines as u16 + 2).min(12)
}

fn draw_input(f: &mut Frame, area: Rect, app: &App) {
    let color = if app.is_generating() { DIM } else { USER };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(color));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let wrapped = wrap(&app.input, inner.width as usize);
    let text: Vec<Line> = wrapped.iter().map(|l| Line::raw(l.clone())).collect();
    f.render_widget(Paragraph::new(text), inner);

    // Place the caret by walking the wrapped lines to the byte offset.
    let (row, col) = caret_position(&app.input, app.cursor, inner.width as usize);
    f.set_cursor_position(Position {
        x: inner.x + col.min(inner.width.saturating_sub(1) as usize) as u16,
        y: inner.y + row.min(inner.height.saturating_sub(1) as usize) as u16,
    });
}

/// Where the caret sits once the input has been wrapped.
fn caret_position(input: &str, cursor: usize, width_cells: usize) -> (usize, usize) {
    let head = &input[..cursor.min(input.len())];
    let lines = wrap(head, width_cells);
    match lines.last() {
        // A caret exactly at a wrap boundary belongs on the next line.
        Some(last) if width(last) >= width_cells => (lines.len(), 0),
        Some(last) => (lines.len().saturating_sub(1), width(last)),
        None => (0, 0),
    }
}

fn draw_footer(f: &mut Frame, area: Rect, app: &App) {
    let hint = if let Some(status) = &app.status {
        Line::styled(format!(" {status}"), Style::default().fg(Color::Yellow))
    } else if app.is_generating() {
        Line::styled(" esc cancel · ctrl+c quit", Style::default().fg(DIM))
    } else {
        Line::styled(
            " enter send · alt+enter newline · pgup/pgdn scroll · ctrl+l clear · ctrl+c quit",
            Style::default().fg(DIM),
        )
    };
    f.render_widget(Paragraph::new(hint), area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caret_tracks_the_end_of_input() {
        assert_eq!(caret_position("hello", 5, 20), (0, 5));
        assert_eq!(caret_position("", 0, 20), (0, 0));
    }

    #[test]
    fn caret_moves_to_the_next_row_at_a_wrap() {
        // Ten cells of input in a ten-cell box: the caret belongs below.
        assert_eq!(caret_position("0123456789", 10, 10), (1, 0));
    }

    #[test]
    fn caret_counts_cjk_as_two_cells() {
        // Three double-width characters is six cells in.
        assert_eq!(caret_position("你好世", "你好世".len(), 20), (0, 6));
    }
}
