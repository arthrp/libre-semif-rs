//! Create-only JSONL command line for direct scoring.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use serde_json::Value;

use crate::error::Error;
use crate::loader::load_model;
use crate::row::validate_row;

#[derive(Parser, Debug)]
#[command(
    name = "libre-semif-rs",
    about = "Score declared options from llama.cpp last-position logits"
)]
struct Args {
    /// Scoring mode. Only direct is implemented.
    #[arg(long)]
    mode: String,
    /// Hugging Face model id or a local directory with the tokenizer and chat template.
    #[arg(long)]
    model: String,
    /// 40-character commit for a remote model, or a revision label for a local directory.
    #[arg(long)]
    revision: String,
    /// Local GGUF checkpoint.
    #[arg(long)]
    gguf: PathBuf,
    /// JSONL decisions. Each line is one row.
    #[arg(long)]
    input: PathBuf,
    /// New JSONL path. Existing files are refused.
    #[arg(long)]
    output: PathBuf,
    /// Maximum prompt tokens. Prompts are never truncated.
    #[arg(long, default_value_t = 4096)]
    max_tokens: i64,
    /// CPU threads for llama.cpp. Defaults to the visible core count.
    #[arg(long)]
    llama_threads: Option<i64>,
}

pub fn run_from<I, S>(args: I) -> ExitCode
where
    I: IntoIterator<Item = S>,
    S: Into<std::ffi::OsString> + Clone,
{
    let args = match Args::try_parse_from(args) {
        Ok(args) => args,
        Err(error) => {
            let code = u8::try_from(error.exit_code()).unwrap_or(2);
            let _ = error.print();
            return ExitCode::from(code);
        }
    };
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            if error.usage {
                ExitCode::from(2)
            } else {
                ExitCode::from(1)
            }
        }
    }
}

fn run(args: Args) -> Result<(), Error> {
    if args.mode != "direct" {
        return Err(Error::usage("--mode accepts only direct"));
    }
    if args.output.exists() || args.max_tokens < 1 {
        return Err(Error::usage(
            "Output must be new and max-tokens must be positive",
        ));
    }
    let threads = match args.llama_threads {
        Some(value) if value < 1 => return Err(Error::usage("--llama-threads must be positive")),
        Some(value) => Some(
            i32::try_from(value).map_err(|_| Error::usage("--llama-threads must be positive"))?,
        ),
        None => None,
    };
    if !args.gguf.is_file() {
        return Err(Error::usage(
            "--gguf must point at an existing GGUF file; requires --gguf",
        ));
    }
    let max_tokens =
        u32::try_from(args.max_tokens).map_err(|_| Error::usage("max-tokens is too large"))?;
    let rows = read_rows(&args.input)?;
    if rows.is_empty() {
        return Err(Error::new("Input is empty"));
    }
    for row in &rows {
        validate_row(row)?;
    }
    let mut session = load_model(&args.model, &args.revision, &args.gguf, threads, max_tokens)?;
    if let Some(parent) = args.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|error| {
                Error::new(format!("Failed to create {}: {error}", parent.display()))
            })?;
        }
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)
        .map_err(|error| {
            Error::new(format!(
                "Output must be new and max-tokens must be positive: {error}"
            ))
        })?;
    for row in &rows {
        let result = session.score(row, max_tokens)?;
        let line = serde_json::to_string(&result)
            .map_err(|error| Error::new(format!("Failed to encode the result: {error}")))?;
        writeln!(output, "{line}")
            .map_err(|error| Error::new(format!("Failed to write the result: {error}")))?;
        output
            .flush()
            .map_err(|error| Error::new(format!("Failed to flush the result: {error}")))?;
    }
    Ok(())
}

fn read_rows(path: &PathBuf) -> Result<Vec<Value>, Error> {
    let file = File::open(path)
        .map_err(|error| Error::new(format!("Failed to read {}: {error}", path.display())))?;
    let mut rows = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line
            .map_err(|error| Error::new(format!("Failed to read {}: {error}", path.display())))?;
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str(&line).map_err(|error| {
            Error::new(format!("Input line {} is not JSON: {error}", index + 1))
        })?;
        rows.push(row);
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("libre-semif-rs-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn base_args(dir: &std::path::Path, extra: &[&str]) -> Vec<String> {
        let mut args = vec![
            "libre-semif-rs".to_string(),
            "--mode".to_string(),
            "direct".to_string(),
            "--model".to_string(),
            "Qwen/Qwen3.5-4B".to_string(),
            "--revision".to_string(),
            "a".repeat(40),
            "--gguf".to_string(),
            dir.join("missing.gguf").display().to_string(),
            "--input".to_string(),
            dir.join("in.jsonl").display().to_string(),
            "--output".to_string(),
            dir.join("out.jsonl").display().to_string(),
        ];
        args.extend(extra.iter().map(|item| (*item).to_string()));
        args
    }

    #[test]
    fn existing_output_is_refused_before_loading() {
        let dir = temp_dir("exists");
        let output = dir.join("out.jsonl");
        fs::write(&output, "keep\n").unwrap();
        let code = run_from(base_args(&dir, &[]));
        assert_eq!(code, ExitCode::from(2));
        assert_eq!(fs::read_to_string(&output).unwrap(), "keep\n");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn zero_max_tokens_and_threads_fail_before_loading() {
        let dir = temp_dir("flags");
        assert_eq!(
            run_from(base_args(&dir, &["--max-tokens", "0"])),
            ExitCode::from(2)
        );
        assert_eq!(
            run_from(base_args(&dir, &["--llama-threads", "0"])),
            ExitCode::from(2)
        );
        assert!(!dir.join("out.jsonl").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_gguf_file_is_rejected() {
        let dir = temp_dir("gguf");
        let code = run_from(base_args(&dir, &[]));
        assert_eq!(code, ExitCode::from(2));
        assert!(!dir.join("out.jsonl").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn other_modes_are_rejected() {
        let dir = temp_dir("mode");
        let mut args = base_args(&dir, &[]);
        let mode = args
            .iter_mut()
            .find(|item| item.as_str() == "direct")
            .unwrap();
        *mode = "serial".to_string();
        assert_eq!(run_from(args), ExitCode::from(2));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_input_fails_before_loading() {
        let dir = temp_dir("empty");
        fs::write(dir.join("in.jsonl"), "\n").unwrap();
        fs::write(dir.join("missing.gguf"), b"gguf").unwrap();
        let code = run_from(base_args(&dir, &[]));
        assert_eq!(code, ExitCode::from(1));
        assert!(!dir.join("out.jsonl").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
