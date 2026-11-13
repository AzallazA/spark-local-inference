/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! GLM think envelopes and raw-string/JSON argument tags from the chat template.

use crate::{
    Result,
    output::tagged::{ToolDialect, parameter_value},
};
use serde_json::{Map, Value};

#[cfg(test)]
use crate::output::tagged::{OutputConfig, TaggedOutput};

pub struct ArgumentTags;
impl ToolDialect for ArgumentTags {
    const NAME: &'static str = "GLM";
    const TRIM_NEWLINES: bool = false;
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";

    fn parse(body: &str, tools: &[Value]) -> Result<Vec<(String, String)>> {
        let body = body.trim();
        let split = body.find("<arg_key>").unwrap_or(body.len());
        let name = body[..split].trim();
        if name.is_empty() {
            return Err("tool function name is empty".into());
        }
        let mut rest = &body[split..];
        let function = tools
            .iter()
            .filter_map(|tool| tool.get("function"))
            .find(|function| function.get("name").and_then(Value::as_str) == Some(name));
        let properties = function
            .and_then(|function| function.get("parameters"))
            .and_then(|parameters| parameters.get("properties"))
            .and_then(Value::as_object);
        let mut arguments = Map::new();
        while !rest.trim().is_empty() {
            let remaining = rest.trim_start();
            let key = remaining.strip_prefix("<arg_key>").ok_or_else(|| {
                // Name what follows instead of echoing generated text into the error.
                let marker = if remaining.starts_with("</arg_value>") {
                    "closing-arg-value"
                } else if remaining.starts_with("</tool_call>") {
                    "closing-tool-call"
                } else if remaining.starts_with("<arg_value>") {
                    "arg-value-without-key"
                } else if remaining.starts_with("<arg_key") {
                    "malformed-arg-key"
                } else if remaining.starts_with('<') {
                    "unexpected-tag"
                } else {
                    "unexpected-text"
                };
                format!(
                    "GLM tool call requires an argument key (remaining_bytes={}, next_marker={marker})",
                    remaining.len()
                )
            })?;
            let (key, tail) = key
                .split_once("</arg_key>")
                .ok_or("unterminated GLM argument key")?;
            // Keys are trimmed as vLLM's GLM-4.7 parser does (`key.strip()`); values stay raw.
            let key = key.trim();
            if key.is_empty() || arguments.contains_key(key) {
                return Err("GLM argument names must be nonempty and unique".into());
            }
            let value = tail
                .trim_start()
                .strip_prefix("<arg_value>")
                .ok_or("GLM argument requires a value tag")?;
            let end = value_end(value).ok_or("unterminated GLM argument value")?;
            let (value, tail) = (&value[..end], &value[end + VALUE_END.len()..]);
            let schema = properties.and_then(|properties| properties.get(key));
            // Off-schema values are retained (parameter_value); only an invalid type declaration fails here.
            let converted = parameter_value(value, schema).map_err(|error| {
                let declared = schema
                    .and_then(|schema| schema.get("type"))
                    .map_or_else(|| "<missing-or-union>".into(), Value::to_string);
                crate::FrontendError::Input(crate::Failure::caused_by(
                    format!(
                        "GLM argument '{key}' (declared type {declared}) failed conversion: {error}"
                    ),
                    error,
                ))
            })?;
            arguments.insert(key.to_owned(), converted);
            rest = tail;
        }
        Ok(vec![(
            name.to_owned(),
            serde_json::to_string(&arguments).map_err(|error| {
                crate::FrontendError::Input(crate::Failure::caused_by(error.to_string(), error))
            })?,
        )])
    }
    fn starts_call(text: &str) -> Option<bool> {
        // Calls continue with the function name and an argument key or closing tag;
        // quoted tags followed by prose remain text.
        let text = text.strip_prefix(Self::OPEN).unwrap_or(text).trim_start();
        if text.is_empty() {
            return None;
        }
        let end = text.find(|c: char| c == '<' || c == '>' || c.is_whitespace())?;
        if end == 0 {
            return Some(false);
        }
        let next = text[end..].trim_start();
        if ["<arg_key>", "</tool_call>"]
            .iter()
            .any(|tag| next.starts_with(tag))
        {
            Some(true)
        } else if ["<arg_key>", "</tool_call>"]
            .iter()
            .any(|tag| tag.starts_with(next))
        {
            None
        } else {
            Some(false)
        }
    }
}

const VALUE_END: &str = "</arg_value>";

/// Byte offset of the `</arg_value>` that closes a value, in the text after `<arg_value>`.
///
/// Values are raw text, so code or XML passed as a string can contain `</arg_value>` itself
/// (vLLM's non-greedy regex splits there). A closing tag ends the value only when, after optional
/// whitespace, the next argument key or the end of the call body follows; the body handed to
/// `parse` stops before `</tool_call>`, so that tag counts as the end. Any other closing tag is
/// value text. A value that literally contains `</arg_value><arg_key>` stays ambiguous and ends
/// there. When no closing tag qualifies, the first one ends the value as before, so a malformed
/// remainder still gets the next-marker diagnostic. Whole calls are parsed only after their
/// closing tag arrives, so streaming never emits a value prefix that a later tag could extend.
fn value_end(value: &str) -> Option<usize> {
    let mut first = None;
    for (at, _) in value.match_indices(VALUE_END) {
        first.get_or_insert(at);
        let next = value[at + VALUE_END.len()..].trim_start();
        if next.is_empty() || next.starts_with("<arg_key>") {
            return Some(at);
        }
    }
    first
}

#[cfg(test)]
type ArgumentTagOutput = TaggedOutput<ArgumentTags>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        output::{Input, ModelOutput, Termination},
        server::output::Accumulator,
    };
    use serde_json::json;

    fn config() -> OutputConfig {
        OutputConfig {
            structured: false,
            start: crate::output::tagged::OutputStart::Reasoning,
            tools: vec![json!({"type":"function", "function": {
            "name":"run", "parameters":{"type":"object", "properties":{
                "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
                "items":{"type":"array"}, "options":{"type":"object"}, "empty":{"type":"null"}
            }}}})],
        }
    }

    #[test]
    fn split_streams_preserve_reasoning_and_decode_typed_tool_arguments() {
        let text = "\nreason α\n</think>\nAnswer\n<tool_call>run<arg_key>text</arg_key><arg_value>\nhello α\n</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>enabled</arg_key><arg_value>true</arg_value><arg_key>items</arg_key><arg_value>[1,2]</arg_value><arg_key>options</arg_key><arg_value>{\"x\":1}</arg_value><arg_key>empty</arg_key><arg_value>null</arg_value></tool_call><tool_call>run</tool_call>\nend";
        for boundary in text
            .char_indices()
            .map(|(index, _)| index)
            .chain([text.len()])
        {
            let mut parser = ArgumentTagOutput::new(config());
            let mut output = Accumulator::new("test".into());
            output
                .apply(parser.push(Input::Text(&text[..boundary])).unwrap())
                .unwrap();
            output
                .apply(parser.push(Input::Text(&text[boundary..])).unwrap())
                .unwrap();
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["reasoning_content"], "\nreason α\n");
            assert_eq!(message["content"], "\nAnswer\n\nend");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[0]["function"]["name"], "run");
            let arguments: Value =
                serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(
                arguments,
                json!({"text":"\nhello α\n", "count":2, "enabled":true, "items":[1,2], "options":{"x":1}, "empty":null})
            );
            assert_eq!(calls[1]["function"]["arguments"], "{}");
        }
    }

    #[test]
    fn malformed_calls_fail() {
        for body in [
            "",
            "run<arg_key></arg_key><arg_value>x</arg_value>",
            "run<arg_key>text</arg_key><arg_value>a</arg_value><arg_key>text</arg_key><arg_value>b</arg_value>",
            "run<arg_key>text</arg_key><arg_value>unfinished",
            "run<arg_key>x</arg_key>text",
        ] {
            assert!(
                ArgumentTags::parse(body, &config().tools).is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn invalid_type_declarations_fail_unless_a_valid_branch_matches() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "bad":{"type":42}, "union":{"anyOf":[{"type":42}]},
            "mixed":{"oneOf":[{"type":42},{"type":"integer"}]}
        }}}})];
        let call = |key: &str, value: &str| {
            ArgumentTags::parse(
                &format!("run<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"),
                &tools,
            )
        };
        for (key, value) in [("bad", "123"), ("union", "123"), ("mixed", "two")] {
            assert!(call(key, value).is_err(), "{key}={value}");
        }
        assert_eq!(call("mixed", "123").unwrap()[0].1, "{\"mixed\":123}");
    }

    #[test]
    fn mismatched_argument_types_retain_generated_values_without_coercion() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "count":{"type":"integer"}, "enabled":{"type":"boolean"},
            "items":{"type":"array"}, "choice":{"anyOf":[{"type":"integer"},{"type":"null"}]},
            "listed":{"anyOf":[{"type":"object"},{"type":"string"}]}
        }}}})];
        let text = "</think><tool_call>run<arg_key>enabled</arg_key><arg_value>\"true\"</arg_value><arg_key>count</arg_key><arg_value>two</arg_value><arg_key>items</arg_key><arg_value>2.5</arg_value><arg_key>choice</arg_key><arg_value>false</arg_value><arg_key>listed</arg_key><arg_value>3</arg_value></tool_call>";
        for split in text.char_indices().map(|(at, _)| at) {
            let mut parser = ArgumentTagOutput::new(OutputConfig {
                tools: tools.clone(),
                ..config()
            });
            let mut output = Accumulator::new("test".into());
            for part in [&text[..split], &text[split..]] {
                output
                    .apply(parser.push(Input::Text(part)).unwrap())
                    .unwrap();
            }
            output
                .apply(parser.finish(Termination::Eos).unwrap())
                .unwrap();
            output.finish().unwrap();
            let message = output.message();
            let arguments: Value = serde_json::from_str(
                message["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                arguments,
                json!({"enabled":"true", "count":"two", "items":2.5, "choice":false, "listed":"3"}),
                "split {split}"
            );
        }
    }

    #[test]
    fn edge_case_values_keep_their_declared_or_generated_form() {
        let tools = vec![json!({"type":"function", "function": {
        "name":"run", "parameters":{"type":"object", "properties":{
            "text":{"type":"string"}, "count":{"type":"integer"}, "enabled":{"type":"boolean"},
            "items":{"type":"array"}, "options":{"type":"object"}, "ratio":{"type":"number"},
            "maybe":{"type":["integer","null"]}
        }}}})];
        let parse = |body: &str| -> Value {
            serde_json::from_str(&ArgumentTags::parse(body, &tools).unwrap()[0].1).unwrap()
        };
        let one = |key: &str, value: &str| -> Value {
            parse(&format!(
                "run<arg_key>{key}</arg_key><arg_value>{value}</arg_value>"
            ))[key]
                .clone()
        };
        // Declared types decode the generated JSON.
        assert_eq!(one("enabled", "true"), json!(true));
        assert_eq!(one("enabled", "false"), json!(false));
        assert_eq!(one("ratio", "2"), json!(2));
        assert_eq!(one("maybe", "null"), Value::Null);
        assert_eq!(one("maybe", "7"), json!(7));
        assert_eq!(
            one("options", r#"{"a":{"b":[1,{"c":null}]},"d":"</x>"}"#),
            json!({"a":{"b":[1,{"c":null}]},"d":"</x>"})
        );
        assert_eq!(
            one("items", r#"[{"k":[1,2]},"s"]"#),
            json!([{"k":[1,2]},"s"])
        );
        // Strings are raw text, never JSON-decoded or trimmed.
        assert_eq!(one("text", "true"), json!("true"));
        assert_eq!(one("text", r#""quoted""#), json!(r#""quoted""#));
        assert_eq!(one("text", ""), json!(""));
        assert_eq!(one("text", " a\nb\t"), json!(" a\nb\t"));
        // Off-schema values are retained without coercion (#179): no comma splitting or wrapping (#59).
        assert_eq!(one("enabled", "True"), json!("True"));
        assert_eq!(one("count", ""), json!(""));
        assert_eq!(one("count", "1.5"), json!(1.5));
        assert_eq!(one("items", "a, b"), json!("a, b"));
        assert_eq!(one("items", r#""[1,2]""#), json!("[1,2]"));
        assert_eq!(one("options", "[1]"), json!([1]));
        // Undeclared keys and undeclared functions keep raw strings.
        assert_eq!(one("extra", "{\"x\":1}"), json!("{\"x\":1}"));
        let unknown =
            ArgumentTags::parse("other<arg_key>n</arg_key><arg_value>3</arg_value>", &tools)
                .unwrap();
        assert_eq!(unknown, vec![("other".into(), "{\"n\":\"3\"}".into())]);
        // Whitespace between tags, as some GLM templates emit, is accepted.
        assert_eq!(
            parse(
                "run\n<arg_key>count</arg_key>\n<arg_value>4</arg_value>\n<arg_key>enabled</arg_key> <arg_value>true</arg_value>\n"
            ),
            json!({"count":4, "enabled":true})
        );
        // A missing required key is the client's to report; the parser does not check `required`.
        assert_eq!(parse("run"), json!({}));
    }

    #[test]
    fn missing_argument_keys_name_the_next_marker_without_echoing_text() {
        for (body, marker) in [
            (
                // A second closing tag before text is value text unless nothing qualifies.
                "run<arg_key>a</arg_key><arg_value>1</arg_value></arg_value> x",
                "closing-arg-value",
            ),
            (
                "run<arg_key>a</arg_key><arg_value>1</arg_value><arg_value>2</arg_value> x",
                "arg-value-without-key",
            ),
            (
                "run<arg_key>a</arg_key><arg_value>1</arg_value><arg_key a>b</arg_key>",
                "malformed-arg-key",
            ),
            (
                "run<arg_key>a</arg_key><arg_value>1</arg_value><tool_call>",
                "unexpected-tag",
            ),
            (
                "run<arg_key>a</arg_key><arg_value>1</arg_value> secret prose",
                "unexpected-text",
            ),
        ] {
            let error = ArgumentTags::parse(body, &config().tools)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(&format!("next_marker={marker}")),
                "{body}: {error}"
            );
            assert!(!error.contains("secret"), "{error}");
        }
        let error = ArgumentTags::parse(
            "run<arg_key>bad</arg_key><arg_value>1</arg_value>",
            &[json!({"type":"function","function":{"name":"run","parameters":{"properties":{"bad":{"type":42}}}}})],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("GLM argument 'bad' (declared type 42)"),
            "{error}"
        );
    }

    #[test]
    fn prose_tool_tag_mentions_remain_answer_text() {
        let call = "<tool_call>run<arg_key>text</arg_key><arg_value>x</arg_value></tool_call>";
        for (text, content, calls) in [
            (
                "</think>The `<tool_call>` tag wraps calls. END".to_owned(),
                "The `<tool_call>` tag wraps calls. END",
                0,
            ),
            (
                "</think>Write <tool_call> or `<tool_call>run` literally.".to_owned(),
                "Write <tool_call> or `<tool_call>run` literally.",
                0,
            ),
            (
                format!("</think>Quote `<tool_call>` here.\n{call}"),
                "Quote `<tool_call>` here.\n",
                1,
            ),
        ] {
            for split in text.char_indices().map(|(at, _)| at) {
                let mut parser = ArgumentTagOutput::new(config());
                let mut output = Accumulator::new("test".into());
                for part in [&text[..split], &text[split..]] {
                    output
                        .apply(parser.push(Input::Text(part)).unwrap())
                        .unwrap();
                }
                output
                    .apply(parser.finish(Termination::Eos).unwrap())
                    .unwrap();
                output.finish().unwrap();
                let message = output.message();
                assert_eq!(message["content"], content, "{text:?} at {split}");
                let tools = message["tool_calls"]
                    .as_array()
                    .map_or(&[][..], Vec::as_slice);
                assert_eq!(tools.len(), calls, "{text:?} at {split}");
                if let Some(tool) = tools.first() {
                    assert_eq!(tool["function"]["name"], "run");
                    assert_eq!(tool["function"]["arguments"], "{\"text\":\"x\"}");
                }
            }
        }
    }

    #[test]
    fn truncation_and_structured_output_keep_protocol_semantics() {
        let mut parser = ArgumentTagOutput::new(config());
        parser
            .push(Input::Text("</think><tool_call>run<arg_key>text</arg_key>"))
            .unwrap();
        assert!(parser.finish(Termination::OutputLimit).unwrap().is_empty());
        let mut parser = ArgumentTagOutput::new(OutputConfig {
            structured: true,
            start: crate::output::tagged::OutputStart::Content,
            ..config()
        });
        assert_eq!(
            parser
                .push(Input::Text("<think>{\"x\":1}</think>"))
                .unwrap(),
            vec![crate::output::Event::Content(
                "<think>{\"x\":1}</think>".into()
            )]
        );
    }

    #[test]
    fn closing_value_tags_inside_values_are_value_text() {
        let parse = |body: &str| -> Value {
            serde_json::from_str(&ArgumentTags::parse(body, &config().tools).unwrap()[0].1).unwrap()
        };
        // A closing tag followed by more text, then the real one before the next key.
        assert_eq!(
            parse(
                "run<arg_key>text</arg_key><arg_value>a</arg_value>b</arg_value><arg_key>count</arg_key><arg_value>1</arg_value>"
            ),
            json!({"text":"a</arg_value>b", "count":1})
        );
        // The live markup_in_value case, as the last argument.
        assert_eq!(
            parse("run<arg_key>text</arg_key><arg_value><a></arg_value><b/></a></arg_value>"),
            json!({"text":"<a></arg_value><b/></a>"})
        );
        // A value ending with the literal tag, right before the real one: at the end and before a key.
        assert_eq!(
            parse("run<arg_key>text</arg_key><arg_value>x</arg_value></arg_value>"),
            json!({"text":"x</arg_value>"})
        );
        assert_eq!(
            parse(
                "run<arg_key>text</arg_key><arg_value>x</arg_value></arg_value>\n<arg_key>count</arg_key><arg_value>3</arg_value>\n"
            ),
            json!({"text":"x</arg_value>", "count":3})
        );
        // Key-looking text inside a value, with and without a stray closing tag.
        assert_eq!(
            parse(
                "run<arg_key>text</arg_key><arg_value>use <arg_key>k</arg_key><arg_value>v</arg_value> tags</arg_value><arg_key>count</arg_key><arg_value>2</arg_value><arg_key>options</arg_key><arg_value>{\"t\":\"<arg_key>\"}</arg_value>"
            ),
            json!({"text":"use <arg_key>k</arg_key><arg_value>v</arg_value> tags", "count":2, "options":{"t":"<arg_key>"}})
        );
        // Typed values keep #179's retention when they contain the tag.
        assert_eq!(
            parse("run<arg_key>count</arg_key><arg_value>1</arg_value>2</arg_value>"),
            json!({"count":"1</arg_value>2"})
        );
        // Keys are trimmed (vLLM strips them); a closing tag then the next key stays ambiguous and ends the value.
        assert_eq!(
            parse(
                "run<arg_key> text\n</arg_key><arg_value> a </arg_value><arg_key>count </arg_key><arg_value>4</arg_value>"
            ),
            json!({"text":" a ", "count":4})
        );
        assert!(
            ArgumentTags::parse(
                "run<arg_key>text</arg_key><arg_value>a</arg_value><arg_key> text</arg_key><arg_value>b</arg_value>",
                &config().tools
            )
            .is_err()
        );
        assert!(
            ArgumentTags::parse(
                "run<arg_key> </arg_key><arg_value>a</arg_value>",
                &config().tools
            )
            .is_err()
        );
    }

    #[test]
    fn streamed_closing_value_tags_in_values_are_held_until_the_call_closes() {
        let text = "</think>Hi\n<tool_call>run<arg_key>text</arg_key><arg_value><a></arg_value><b/></arg_value></arg_value>\n<arg_key> count</arg_key><arg_value>5</arg_value><arg_key>options</arg_key><arg_value>{\"k\":\"</arg_value>x<arg_key>\"}</arg_value></tool_call>\nend";
        let expected = json!({"text":"<a></arg_value><b/></arg_value>", "count":5, "options":{"k":"</arg_value>x<arg_key>"}});
        let check = |parts: &[&str], label: &str| {
            let mut parser = ArgumentTagOutput::new(config());
            let mut output = Accumulator::new("test".into());
            let mut arguments = String::new();
            let mut events = vec![];
            for part in parts {
                events.extend(parser.push(Input::Text(part)).unwrap());
            }
            events.extend(parser.finish(Termination::Eos).unwrap());
            for event in &events {
                if let crate::output::Event::ToolArguments { text, .. } = event {
                    arguments.push_str(text);
                }
            }
            output.apply(events).unwrap();
            output.finish().unwrap();
            let message = output.message();
            assert_eq!(message["content"], "Hi\n\nend", "{label}");
            let calls = message["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 1, "{label}");
            // The streamed deltas are exactly the final arguments: nothing lost or repeated.
            assert_eq!(
                calls[0]["function"]["arguments"],
                arguments.as_str(),
                "{label}"
            );
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                expected,
                "{label}"
            );
        };
        let boundaries: Vec<_> = text
            .char_indices()
            .map(|(at, _)| at)
            .chain([text.len()])
            .collect();
        for &split in &boundaries {
            check(&[&text[..split], &text[split..]], &format!("split {split}"));
        }
        // Three pieces: a cut inside each `</arg_value>` combined with a cut inside the following tag.
        for (closing, _) in text.match_indices("</arg_value>") {
            for inner in closing + 1..closing + "</arg_value>".len() {
                for after in [inner + 13, inner + 15, inner + 18] {
                    if after < text.len() && text.is_char_boundary(after) {
                        check(
                            &[&text[..inner], &text[inner..after], &text[after..]],
                            &format!("cuts {inner}/{after}"),
                        );
                    }
                }
            }
        }
        let chars: Vec<_> = text
            .char_indices()
            .map(|(at, c)| &text[at..at + c.len_utf8()])
            .collect();
        check(&chars, "char by char");
    }
}
