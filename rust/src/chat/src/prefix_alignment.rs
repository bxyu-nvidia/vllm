// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;

use vllm_text::Prompt;
use vllm_tokenizer::Tokenizer;

use crate::capture::CaptureAlignment;
use crate::error::{Result, bail_invalid_prefix_alignment, invalid_prefix_alignment};
use crate::{ChatRenderer, ChatRequest, ChatRole, GenerationPromptMode, RenderedPrompt};

pub(crate) fn validate_request(request: &ChatRequest, enabled: bool) -> Result<()> {
    let Some(prefix) = &request.required_prefix_token_ids else {
        return Ok(());
    };
    if !enabled {
        bail_invalid_prefix_alignment!(
            "start the Rust frontend with --enable-prefix-token-alignment"
        );
    }
    if prefix.is_empty() {
        return Ok(());
    }
    if request.has_multimodal() {
        bail_invalid_prefix_alignment!("multimodal prefix alignment is not supported");
    }
    if request.prompt_truncation.is_some() {
        bail_invalid_prefix_alignment!(
            "prompt truncation cannot preserve an exact required prefix"
        );
    }
    if request.chat_options.generation_prompt_mode == GenerationPromptMode::ContinueFinalAssistant {
        bail_invalid_prefix_alignment!(
            "continue_final_message is not supported with an exact prior prefix"
        );
    }
    if !request.messages.iter().any(|message| message.role() == ChatRole::Assistant) {
        bail_invalid_prefix_alignment!("a nonempty prefix requires a prior assistant message");
    }
    Ok(())
}

/// Render only through the prior assistant turn to locate its boundary in the
/// full template. Keep the model's tokens even when history rendering changes
/// whitespace, tool formatting, or removes reasoning from previous turns.
pub(crate) fn align_prompt(
    request: &ChatRequest,
    rendered: &mut RenderedPrompt,
    required_prefix: &[u32],
    renderer: &dyn ChatRenderer,
    tokenizer: &dyn Tokenizer,
    eos_ids: &BTreeSet<u32>,
) -> Result<Option<CaptureAlignment>> {
    let boundary = locate_boundary(request, rendered, renderer, tokenizer, eos_ids)?;
    let Prompt::TokenIds(full_tokens) = std::mem::take(&mut rendered.prompt) else {
        unreachable!("locate_boundary tokenizes the prompt");
    };
    let (tokens, boundary) = splice_at_boundary(required_prefix, full_tokens, boundary);
    rendered.prompt = Prompt::TokenIds(tokens);
    Ok(Some(boundary))
}

/// Locate a continuation without fetching or transmitting the stored parent.
pub(crate) fn locate_boundary(
    request: &ChatRequest,
    rendered: &mut RenderedPrompt,
    renderer: &dyn ChatRenderer,
    tokenizer: &dyn Tokenizer,
    eos_ids: &BTreeSet<u32>,
) -> Result<CaptureAlignment> {
    let last_assistant = request
        .messages
        .iter()
        .rposition(|message| message.role() == ChatRole::Assistant)
        .ok_or_else(|| {
            invalid_prefix_alignment!("a nonempty prefix requires a prior assistant message")
        })?;
    let mut history = request.clone();
    history.messages.truncate(last_assistant + 1);
    history.chat_options.generation_prompt_mode = GenerationPromptMode::NoGenerationPrompt;
    let prefix_rendered = renderer.render(&history)?;
    let encode = |prompt: Prompt| -> Result<Vec<u32>> {
        Ok(match prompt {
            Prompt::Text(text) => tokenizer.encode(&text, request.add_special_tokens)?,
            Prompt::TokenIds(ids) => ids,
        })
    };
    let template_prefix = encode(prefix_rendered.prompt)?;
    let full_tokens = encode(std::mem::take(&mut rendered.prompt))?;
    let boundary = find_boundary(&template_prefix, &full_tokens, eos_ids)?;
    rendered.prompt = Prompt::TokenIds(full_tokens);
    Ok(boundary)
}

#[cfg(test)]
fn splice_prefix(
    required_prefix: &[u32],
    template_prefix: &[u32],
    full_tokens: Vec<u32>,
    eos_ids: &BTreeSet<u32>,
) -> Result<(Vec<u32>, Option<CaptureAlignment>)> {
    if required_prefix.is_empty() {
        return Ok((full_tokens, None));
    }
    let alignment = find_boundary(template_prefix, &full_tokens, eos_ids)?;
    let (tokens, alignment) = splice_at_boundary(required_prefix, full_tokens, alignment);
    Ok((tokens, Some(alignment)))
}

fn find_boundary(
    template_prefix: &[u32],
    full_tokens: &[u32],
    eos_ids: &BTreeSet<u32>,
) -> Result<CaptureAlignment> {
    if eos_ids.is_empty() {
        bail_invalid_prefix_alignment!("the model/tokenizer must define an EOS token ID");
    }
    // Positions can shift when a template strips historical reasoning. The
    // ordinal of the completed turn's EOS remains the same in both renders.
    let turn_end_count = template_prefix.iter().filter(|id| eos_ids.contains(id)).count();
    if turn_end_count == 0 {
        bail_invalid_prefix_alignment!("the rendered assistant history contains no EOS boundary");
    }
    let boundary = full_tokens
        .iter()
        .enumerate()
        .filter(|(_, id)| eos_ids.contains(id))
        .nth(turn_end_count - 1)
        .map(|(index, _)| index)
        .ok_or_else(|| {
            invalid_prefix_alignment!(
                "EOS boundary #{turn_end_count} is missing from the full prompt"
            )
        })?;
    Ok(CaptureAlignment {
        suffix_start: boundary + 1,
        turn_end_token_id: full_tokens[boundary],
        eos_token_ids: eos_ids.iter().copied().collect(),
    })
}

fn splice_at_boundary(
    required_prefix: &[u32],
    full_tokens: Vec<u32>,
    mut alignment: CaptureAlignment,
) -> (Vec<u32>, CaptureAlignment) {
    let boundary = alignment.suffix_start - 1;
    let mut aligned = Vec::with_capacity(required_prefix.len() + full_tokens.len() - boundary);
    aligned.extend_from_slice(required_prefix);
    // A length-limited generation has no EOS. Insert the template's terminator;
    // otherwise retain the exact EOS emitted by the model, including alternate IDs.
    if !required_prefix.last().is_some_and(|id| alignment.eos_token_ids.contains(id)) {
        aligned.push(full_tokens[boundary]);
    }
    alignment.suffix_start = aligned.len();
    aligned.extend_from_slice(&full_tokens[boundary + 1..]);
    (aligned, alignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatMessage;

    #[test]
    fn preserves_generated_reasoning_and_retokenized_tool_tokens() {
        let result = splice_prefix(
            &[10, 2, 90, 91, 92, 2],
            &[10, 2, 30, 31, 32, 2],
            vec![10, 2, 32, 2, 40, 2, 50],
            &BTreeSet::from([2]),
        )
        .unwrap();
        assert_eq!(result.0, [10, 2, 90, 91, 92, 2, 40, 2, 50]);
    }

    #[test]
    fn length_limited_output_gets_template_terminator() {
        let result = splice_prefix(
            &[10, 90],
            &[10, 30, 2],
            vec![10, 30, 2, 40],
            &BTreeSet::from([2]),
        )
        .unwrap();
        assert_eq!(result.0, [10, 90, 2, 40]);
        let boundary = result.1.unwrap();
        assert_eq!(boundary.suffix_start, 3);
        assert_eq!(boundary.turn_end_token_id, 2);
        assert_eq!(boundary.eos_token_ids, vec![2]);
    }

    #[test]
    fn retains_alternate_model_eos_instead_of_template_eos() {
        let result = splice_prefix(
            &[10, 90, 3],
            &[10, 30, 2],
            vec![10, 30, 2, 40],
            &BTreeSet::from([2, 3]),
        )
        .unwrap();
        assert_eq!(result.0, [10, 90, 3, 40]);
        let boundary = result.1.unwrap();
        assert_eq!(boundary.suffix_start, 3);
        assert_eq!(boundary.turn_end_token_id, 2);
        assert_eq!(boundary.eos_token_ids, vec![2, 3]);
    }

    #[test]
    fn empty_prefix_keeps_normal_prompt_without_requiring_eos() {
        assert_eq!(
            splice_prefix(&[], &[], vec![10, 20], &BTreeSet::new()).unwrap(),
            (vec![10, 20], None)
        );
    }

    #[test]
    fn missing_boundaries_fail_as_request_errors() {
        for (history, full, eos) in [
            (vec![10, 2], vec![10, 2, 40], BTreeSet::new()),
            (vec![10], vec![10, 2, 40], BTreeSet::from([2])),
            (vec![10, 2, 20, 2], vec![10, 2, 40], BTreeSet::from([2])),
        ] {
            assert!(
                splice_prefix(&[90, 2], &history, full, &eos)
                    .unwrap_err()
                    .is_request_validation_error()
            );
        }
    }

    #[test]
    fn alignment_requires_opt_in_and_completed_assistant_history() {
        let mut request = ChatRequest::for_test();
        request.required_prefix_token_ids = Some(vec![90, 2]);
        assert!(validate_request(&request, false).is_err());
        assert!(validate_request(&request, true).is_err());
        request.messages.insert(0, ChatMessage::text(ChatRole::Assistant, "previous"));
        validate_request(&request, true).unwrap();
        request.chat_options.generation_prompt_mode = GenerationPromptMode::ContinueFinalAssistant;
        assert!(validate_request(&request, true).is_err());
    }
    #[test]
    fn jinja_alignment_preserves_reasoning_and_tool_history() {
        use crate::renderer::hf::HfChatRenderer;
        use vllm_tokenizer::test_utils::TestTokenizer;
        let renderer = HfChatRenderer::new(Some(
            "{% for m in messages %}{{ m.role }}: {% if m.role == 'assistant' and messages[-1].role == 'tool' %}stripped{% else %}{{ m.content }}{% endif %}<eos>{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}".into()
        ), Default::default(), Default::default()).unwrap();
        let tokenizer = TestTokenizer::new().with_special_token("<eos>", 256);
        let request: ChatRequest = ChatRequest {
            messages: vec![
                ChatMessage::text(ChatRole::User, "hi"),
                ChatMessage::text(ChatRole::Assistant, "reasoning and tool call"),
                ChatMessage::tool_response(
                    crate::ChatContent::Text("tool result".into()),
                    "call_1",
                ),
            ],
            ..ChatRequest::for_test()
        };
        let prefix = tokenizer
            .encode(
                "user: hi<eos>assistant: EXACT reasoning and tool call<eos>",
                false,
            )
            .unwrap();
        let mut rendered = renderer.render(&request).unwrap();
        let boundary = locate_boundary(
            &request,
            &mut rendered,
            &renderer,
            &tokenizer,
            &BTreeSet::from([256]),
        )
        .unwrap();
        let Prompt::TokenIds(tokens) = &rendered.prompt else {
            panic!("expected tokens")
        };
        assert_eq!(tokens[boundary.suffix_start - 1], 256);
        assert_eq!(
            tokens[boundary.suffix_start..],
            tokenizer.encode("tool: tool result<eos>assistant:", false).unwrap()
        );
        align_prompt(
            &request,
            &mut rendered,
            &prefix,
            &renderer,
            &tokenizer,
            &BTreeSet::from([256]),
        )
        .unwrap();
        let mut expected = prefix;
        expected.extend(tokenizer.encode("tool: tool result<eos>assistant:", false).unwrap());
        assert_eq!(rendered.prompt, Prompt::TokenIds(expected));
    }
}
