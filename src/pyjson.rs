//! Python `json.dumps(..., ensure_ascii=False)` for prompt payloads.
//!
//! Object key order is the order stored on [`serde_json::Value`] (`preserve_order`).
//! Floats follow CPython's `repr` for the values exercised by the tests: Rust's
//! debug float is post-processed so exponents use a sign and two digits (`1e+21`,
//! `1e-07`).

use serde_json::{Map, Number, Value};

use crate::error::Error;

pub fn dumps(value: &Value) -> Result<String, Error> {
    let mut out = String::new();
    write_value(&mut out, value)?;
    Ok(out)
}

fn write_value(out: &mut String, value: &Value) -> Result<(), Error> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => write_number(out, number)?,
        Value::String(text) => write_string(out, text),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_value(out, item)?;
            }
            out.push(']');
        }
        Value::Object(map) => write_object(out, map)?,
    }
    Ok(())
}

fn write_object(out: &mut String, map: &Map<String, Value>) -> Result<(), Error> {
    out.push('{');
    for (index, (key, value)) in map.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        write_string(out, key);
        out.push_str(": ");
        write_value(out, value)?;
    }
    out.push('}');
    Ok(())
}

fn write_number(out: &mut String, number: &Number) -> Result<(), Error> {
    if let Some(value) = number.as_i64() {
        out.push_str(&value.to_string());
        return Ok(());
    }
    if let Some(value) = number.as_u64() {
        out.push_str(&value.to_string());
        return Ok(());
    }
    let value = number.as_f64().ok_or_else(non_finite)?;
    if !value.is_finite() {
        return Err(non_finite());
    }
    out.push_str(&python_float(value));
    Ok(())
}

fn non_finite() -> Error {
    Error::new("state must be finite JSON-compatible data")
}

/// Format a finite float the way CPython's `json.dumps` does for common values.
pub fn python_float(value: f64) -> String {
    let raw = format!("{value:?}");
    let Some(pos) = raw.find('e') else {
        return raw;
    };
    let (mantissa, exponent) = raw.split_at(pos);
    let exponent = &exponent[1..];
    let (sign, digits) = if let Some(digits) = exponent.strip_prefix('-') {
        ('-', digits)
    } else if let Some(digits) = exponent.strip_prefix('+') {
        ('+', digits)
    } else {
        ('+', exponent)
    };
    let magnitude: i32 = digits.parse().unwrap_or(0);
    format!("{mantissa}e{sign}{magnitude:02}")
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if (ch as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", u32::from(ch)));
            }
            ch => out.push(ch),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_matches_python_json() {
        let samples = [
            (json!("café"), r#""café""#),
            (
                json!({
                    "evidence": "owned evidence",
                    "criterion": "Which answer follows?",
                    "options": [
                        {"letter": "A", "description": "Yes."},
                        {"letter": "B", "description": "No."}
                    ]
                }),
                r#"{"evidence": "owned evidence", "criterion": "Which answer follows?", "options": [{"letter": "A", "description": "Yes."}, {"letter": "B", "description": "No."}]}"#,
            ),
            (
                json!({"policy": "Never request passwords", "candidate": ["invoice id"]}),
                r#"{"policy": "Never request passwords", "candidate": ["invoice id"]}"#,
            ),
            (json!([1, 2, 3]), "[1, 2, 3]"),
            (json!("line\n\t\"\\"), r#""line\n\t\"\\""#),
            (
                json!({"b": true, "n": null, "z": []}),
                r#"{"b": true, "n": null, "z": []}"#,
            ),
        ];
        for (value, expected) in samples {
            assert_eq!(dumps(&value).unwrap(), expected);
        }
    }

    #[test]
    fn floats_match_cpython_repr() {
        let parsed =
            serde_json::from_str::<Value>(r#"{"n":1,"f":1.0,"g":0.1,"h":-2.5,"i":1e21,"j":1e-7}"#)
                .unwrap();
        assert_eq!(
            dumps(&parsed).unwrap(),
            r#"{"n": 1, "f": 1.0, "g": 0.1, "h": -2.5, "i": 1e+21, "j": 1e-07}"#
        );
        assert_eq!(python_float(1e16), "1e+16");
        assert_eq!(python_float(1e-5), "1e-05");
    }

    #[test]
    fn json_cannot_carry_nan() {
        assert!(Number::from_f64(f64::NAN).is_none());
        assert!(Number::from_f64(f64::INFINITY).is_none());
    }
}
